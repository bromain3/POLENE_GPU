"""
mixtureOfExperts.py
───────────────────
A `_wrapper` subclass that holds a list of expert networks (any `_wrapper`
subclass) and a fitted K-Means clusterer.

At inference time each sample is routed to its nearest cluster centroid using
a dedicated `clustering_scaler` (the scaler that was used to fit K-Means),
then each sub-batch is processed by its own expert with its own input/output
scalers.  This makes the wrapper fully general: experts may have different
scalers without any assumption of a shared one.

Usage example
─────────────
    from POLENE_GPU.NN.models.moeDataPreparation import prepare_moe_datasets
    from POLENE_GPU.NN.models.mixtureOfExperts   import MixtureOfExperts
    from POLENE_GPU.NN.models.resNet             import ResNet
    from POLENE_GPU.NN.trainer.baseTrainer       import baseTrainer
    from POLENE_GPU.preProcessing.scaler.stdScaler import stdScaler

    # 1. Split the dataset — clustering_scaler is now returned
    clusterer, clustering_scaler, X_list, Y_list, scaler_list = prepare_moe_datasets(
        Xdtb, Ydtb, stdScaler(), n_clusters=4
    )

    # 2. Train one expert per cluster
    experts = []
    for k, (X_sub, Y_sub, scaler) in enumerate(zip(X_list, Y_list, scaler_list)):
        X_scaled = scaler.transform(X_sub)
        Y_scaled = scaler.transform(Y_sub)

        model = ResNet(X_sub.nFeatures, 54, 2, 350, layerNorm=True)
        model.inputScaler  = scaler
        model.outputScaler = scaler

        trainer = baseTrainer(...)
        trainer.train(X_scaled, Y_scaled, model)
        experts.append(model)

    # 3. Wrap into MoE — pass the clustering_scaler explicitly
    moe = MixtureOfExperts(experts, clusterer, clustering_scaler)

    # 4. Save / load — uses _wrapper's save/load directly
    moe.save("checkpoints/", "moe.pt")
    moe2 = MixtureOfExperts.load("checkpoints/moe.pt")

    # 5. Forward — same interface as any _wrapper (inherited, untouched)
    Y_pred = moe(X_serie)                      # serie in → serie out
    Y_pred = moe(x_tensor, tensorInput=True)   # tensor in → tensor out
"""

import importlib
import pickle

import torch
import torch.nn as nn

from POLENE_GPU.NN.models._wrapper import _wrapper
from POLENE_GPU.preProcessing.scaler.identityScaler import identityScaler


class MixtureOfExperts(_wrapper):
    """
    Mixture-of-Experts wrapper.

    Parameters
    ----------
    experts            : list[_wrapper]
        One trained expert per cluster.  Each expert must expose
        `.inputScaler`, `.outputScaler`, and `.torchForward(x)`.
    clusterer          : fitted sklearn KMeans
    clustering_scaler  : fitted scaler instance
        The scaler that was used to produce the space in which K-Means was
        run.  Used exclusively for routing — never for the expert predictions.
        Defaults to identityScaler() (raw-input routing).
    """

    def __init__(self, experts: list, clusterer, clustering_scaler=None):
        nn.Module.__init__(self)
        _wrapper.__init__(self)

        if not experts:
            raise ValueError("experts list must be non-empty")

        self.experts           = nn.ModuleList(experts)
        self.clusterer         = clusterer
        self.n_experts         = len(experts)
        self.clustering_scaler = clustering_scaler if clustering_scaler is not None else identityScaler()

        # Cluster centres as a buffer so .to(device) moves them automatically
        centres = torch.tensor(
            clusterer.cluster_centers_, dtype=torch.float32
        )                               # (K, F_scaled)
        self.register_buffer("_centres", centres)

        # MoE-level identity scalers so that validators calling
        # model.inputScaler.transform / model.outputScaler.inverse_transform
        # pass X through unchanged before torchForward handles everything.
        self.inputScaler  = identityScaler()
        self.outputScaler = identityScaler()

    # ──────────────────────────────────────────────────────────────────────
    # Core routing logic — forward() itself is inherited from _wrapper
    # ──────────────────────────────────────────────────────────────────────

    def torchForward(self, x: torch.Tensor) -> torch.Tensor:  # TODO: travailler sur le fait que en usage autoregressif, on fait de la modification en place et ça peut poser problème
        """
        Full pipeline per sample:
          1. Scale x with clustering_scaler  →  assign cluster
          2. For each cluster k:
               a. Scale x_sub with expert_k.inputScaler
               b. Run expert_k.torchForward
               c. Inverse-scale with expert_k.outputScaler
          3. Re-assemble in original order
        """
        x_for_routing = self.clustering_scaler.transformTorch(x)   # (N, F_routing)
        labels        = self._assign_clusters(x_for_routing)        # (N,)

        output = torch.empty_like(x)

        for k, expert in enumerate(self.experts):
            mask = (labels == k)
            if not mask.any():
                continue

            x_sub    = x[mask]
            x_scaled = expert.inputScaler.transformTorch(x_sub)
            y_scaled = expert(x_scaled, tensorInput=True)
            output[mask] = expert.outputScaler.inverse_transformTorch(y_scaled)

        return output

    def _assign_clusters(self, x_scaled: torch.Tensor) -> torch.Tensor:
        """Nearest-centroid lookup in PyTorch — runs on whatever device x is on."""
        centres = self._centres.to(x_scaled.device, x_scaled.dtype)  # (K, F)
        dists   = torch.sum(
            (x_scaled.unsqueeze(1) - centres.unsqueeze(0)) ** 2, dim=-1
        )                                                              # (N, K)
        return torch.argmin(dists, dim=-1)                            # (N,)

    def predict_cluster(self, serie_or_tensor, tensorInput: bool = False) -> torch.Tensor:
        """
        Return the cluster index for each sample without running the experts.

        Parameters
        ----------
        serie_or_tensor : serie or torch.Tensor  — raw (unscaled) inputs
        tensorInput     : bool

        Returns
        -------
        labels : torch.LongTensor, shape (N,)
        """
        x = serie_or_tensor if tensorInput else serie_or_tensor.data
        x_for_routing = self.clustering_scaler.transformTorch(x)
        return self._assign_clusters(x_for_routing)

    # ──────────────────────────────────────────────────────────────────────
    # Config / Save / Load — built on top of _wrapper's mechanism
    # ──────────────────────────────────────────────────────────────────────
    # _wrapper.save() pickles self.inputScaler/self.outputScaler (the two
    # identityScalers — harmless) and self._get_config(). The only piece
    # _wrapper doesn't know how to serialize is the *list of expert
    # configs/scalers* plus the clusterer/clustering_scaler, so we fold all
    # of that into _get_config() and rebuild it in _from_config().

    def _get_config(self):
        expert_payloads = []
        for expert in self.experts:
            expert_payloads.append({
                "cls_module": type(expert).__module__,
                "cls_name":   type(expert).__qualname__,
                "config":     expert._get_config(),
                "scalers": {
                    "input":  pickle.dumps(expert.inputScaler),
                    "output": pickle.dumps(expert.outputScaler),
                },
            })

        return {
            "expert_payloads":   expert_payloads,
            "clusterer":         pickle.dumps(self.clusterer),
            "clustering_scaler": pickle.dumps(self.clustering_scaler),
        }

    @classmethod
    def _from_config(cls, config):
        clusterer         = pickle.loads(config["clusterer"])
        clustering_scaler = pickle.loads(config["clustering_scaler"])

        experts = []
        for payload in config["expert_payloads"]:
            mod        = importlib.import_module(payload["cls_module"])
            expert_cls = mod
            for part in payload["cls_name"].split("."):
                expert_cls = getattr(expert_cls, part)

            expert = expert_cls._from_config(payload["config"])
            expert.inputScaler  = pickle.loads(payload["scalers"]["input"])
            expert.outputScaler = pickle.loads(payload["scalers"]["output"])
            experts.append(expert)

        return cls(experts, clusterer, clustering_scaler)

    # save()/load() are fully inherited from _wrapper — no override needed.

    # ──────────────────────────────────────────────────────────────────────
    # Misc
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self):
        lines = [f"MixtureOfExperts(n_experts={self.n_experts})"]
        for i, expert in enumerate(self.experts):
            lines.append(f"  [{i}] {type(expert).__name__}")
        return "\n".join(lines)
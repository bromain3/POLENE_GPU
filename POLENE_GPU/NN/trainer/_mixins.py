"""
Trainer mixins — reusable building blocks with no training state of their own.

Each mixin is self-contained and can be combined freely via multiple inheritance.
All mixins assume the host class has at minimum:
    self.trainingDevice, self.trainingDtype,
    self.warmupSteps, self.decayRate, self.decaySteps  (for SchedulerMixin)
    self.testSize, self.batchSize                       (for SplitMixin / ValidationMixin)
"""

import torch
import torch.nn as nn
import copy
import random
from collections import defaultdict
from POLENE_GPU.NN.trainer._losses import directional_sobolev_losses
from torch.autograd.functional import jacobian
from torch.func import functional_call


# ─────────────────────────────────────────────────────────────────────────────
class SchedulerMixin:
    """Provides the warmup + exponential decay LR schedule."""

    def warmup_exp_decay(self, k: int) -> float:
        if self.warmupSteps > 0 and k < self.warmupSteps:
            return (k + 1) / self.warmupSteps
        return self.decayRate ** ((k - self.warmupSteps) / self.decaySteps)


# ─────────────────────────────────────────────────────────────────────────────
class SplitMixin:
    """Train/test split with optional extra tensors (e.g. Jacobians)."""

    def trainTestSplit(self, X, Y, *extras):
        """
        Split X, Y and any number of extra datasets (e.g. J) with the same
        random permutation.

        Args:
            X, Y    : datasets with a .nSamples attribute and __getitem__
            *extras : any additional datasets that must be split identically

        Returns:
            X_train, X_test, Y_train, Y_test [, extra_train, extra_test, ...]
        """
        all_ds = (X, Y) + extras
        n_samples = X.nSamples
        for ds in all_ds:
            if ds.nSamples != n_samples:
                raise ValueError(
                    "All datasets passed to trainTestSplit must have the same number of samples."
                )

        n_test = int(n_samples * self.testSize)
        perm = torch.randperm(n_samples)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

        result = []
        for ds in all_ds:
            result.append(ds[train_idx].contiguous())
            result.append(ds[test_idx].contiguous())

        return tuple(result)


# ─────────────────────────────────────────────────────────────────────────────
class ValidationMixin:
    """Standard MSE validation loop + optimizer/scheduler factory."""

    def _validation_phase(self, Xval, Yval, model, sampleWeights=None) -> float:
        """
        Evaluate model on (Xval, Yval) with MSE.  Updates self.stats.valLoss.

        Returns:
            mean validation loss (float)
        """
        model.eval()
        total = 0.0

        with torch.no_grad():
            for start in range(0, Xval.nSamples, self.batchSize):
                bX = Xval[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
                bY = Yval[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
                preds = model(bX)

                # --- CALCUL DE LA LOSS PONDÉRÉE ---
                if sampleWeights is not None:
                    # On récupère le batch de poids
                    bW = sampleWeights[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
                    # On calcule la MSE par élément sans réduction
                    raw_loss = nn.functional.mse_loss(preds.data, bY.data, reduction='none')
                    # On applique les poids et on fait la moyenne
                    # bW doit avoir une forme compatible pour le broadcast (ex: [batch, 1] ou [batch])
                    loss = (raw_loss * bW.data).mean()
                else:
                    loss = nn.functional.mse_loss(preds.data, bY.data)

                total += loss.item() * bX.data.size(0)

        mean = total / Xval.nSamples
        self.stats.valLoss.append(mean)
        return mean

    def _validation_phase_with_jacobian(self, Xval, Yval, Jval, model):
        model.eval()
        totalMSE = 0.0
        totalJacobian = 0.0

        with torch.no_grad():
            for start in range(0, Xval.nSamples, self.batchSize):
                bX = Xval[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
                bY = Yval[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
                bJ = Jval[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

                lossMSE, lossJac = directional_sobolev_losses(model, bX, bY, bJ)  # TODO: essayer de désactiver torch.no_grad() pour le calcul autograd de la jvp ...
                totalMSE += lossMSE.item() * bX.data.size(0)
                totalJacobian += lossJac.item() * bX.data.size(0)


        meanMSE = totalMSE / Xval.nSamples
        totalJacobian = totalJacobian / Xval.nSamples


        self.stats.valLoss.append(meanMSE)
        return meanMSE, totalJacobian


    def _build_optimizer(self, model, beta1=None, beta2=None):
        """
        Build the standard Adam optimizer + LambdaLR scheduler pair.

        Returns:
            (optimizer, lrScheduler)
        """

        if beta1 is None:
            beta1 = 0.9
        if beta2 is None:
            beta2 = 0.999

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            betas=(beta1, beta2)
        )


        #optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[self.warmup_exp_decay]
        )
        return optimizer, scheduler


# ─────────────────────────────────────────────────────────────────────────────
class GradNormMixin:
    """
    GradNorm utilities for automatic multi-task loss balancing.

    The host class must call _init_gradnorm() before training to initialise
    the learnable weights and EMA state.

    Typical usage in a trainer:
        self._init_gradnorm(n_tasks=2, device=..., dtype=..., lr=1e-3, alpha=0.5, ema_beta=0.99)
        ...
        w = self._gradnorm_step(optimizer, [weighted_l1, weighted_l2], [raw_l1, raw_l2], model)
    """

    def _init_gradnorm(
        self,
        n_tasks: int,
        device: torch.device,
        dtype: torch.dtype,
        lr: float = 1e-3,
        alpha: float = 0.5,
        ema_beta: float = 0.99,
    ):
        self._gn_alpha = alpha
        self._gn_ema_beta = ema_beta
        self._gn_ema: list[torch.Tensor | None] = [None] * n_tasks

        self._gn_weights = nn.Parameter(
            torch.ones(n_tasks, device=device, dtype=dtype)
        )
        self._gn_weight_optim = torch.optim.Adam([self._gn_weights], lr=lr)

    # ── core helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _grad_norm(loss: torch.Tensor, params: list) -> torch.Tensor:
        """L2 norm of the gradient of `loss` w.r.t. `params`."""
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, create_graph=True
        )
        return torch.norm(
            torch.stack([g.norm() for g in grads if g is not None])
        )

    def _update_ema(self, raw_losses: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Update per-task EMA and return the relative training-rate ratios r_i.

        r_i = ema_i / mean(ema)   — used as the target relative gradient norm.
        """
        beta = self._gn_ema_beta
        for i, loss in enumerate(raw_losses):
            val = loss.detach()
            if self._gn_ema[i] is None:
                self._gn_ema[i] = val
            else:
                self._gn_ema[i] = beta * self._gn_ema[i] + (1.0 - beta) * val

        mean = sum(self._gn_ema) / len(self._gn_ema) + 1e-12  # type: ignore[arg-type]
        return [e / mean for e in self._gn_ema]                # type: ignore[operator]

    def _softmax_weights(self) -> torch.Tensor:
        return torch.softmax(self._gn_weights, dim=0)

    def _gradnorm_step(
        self,
        model_optimizer: torch.optim.Optimizer,
        weighted_losses: list[torch.Tensor],
        raw_losses: list[torch.Tensor],
        model: nn.Module,
        clip_grad: float | None = None,
    ) -> torch.Tensor:
        """
        Full GradNorm update:
          1. Compute per-task gradient norms G_i.
          2. Compute training-rate ratios r_i via EMA.
          3. Back-prop the GradNorm loss into self._gn_weights.
          4. Back-prop the weighted task loss into model parameters.

        Args:
            model_optimizer  : the model's Adam (or other) optimizer
            weighted_losses  : list of w_i * L_i  (already multiplied by softmax weights)
            raw_losses       : list of raw L_i     (for EMA update)
            model            : nn.Module
            clip_grad        : if not None, clip gradient norm of model params

        Returns:
            combined loss (sum of weighted_losses), already back-propped
        """
        shared_params = [p for p in model.parameters() if p.requires_grad]
        G = [self._grad_norm(wl, shared_params) for wl in weighted_losses]
        G_avg = sum(G) / len(G)

        ratios = self._update_ema(raw_losses)

        grad_loss = sum(
            torch.abs(g - G_avg * (r ** self._gn_alpha))
            for g, r in zip(G, ratios)
        )

        self._gn_weight_optim.zero_grad()
        grad_loss.backward(retain_graph=True)
        self._gn_weight_optim.step()

        loss = sum(weighted_losses)
        model_optimizer.zero_grad()
        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        model_optimizer.step()

        return loss



class PCGradMixin:
    def __init__(self, optimizer):
        self._optim = optimizer

    @property
    def param_groups(self):
        return self._optim.param_groups

    def step(self):
        return self._optim.step()

    def zero_grad(self):
        return self._optim.zero_grad()

    def pc_backward(self, objectives):
        """
        Calcule les gradients projetés pour une liste d'objectifs (losses).
        objectives: liste de tensors [loss_v, loss_j]
        """
        params = [p for group in self._optim.param_groups for p in group['params'] if p.requires_grad]

        # 1. Calculer les gradients séparés pour chaque tâche
        grads_per_task = []
        for obj in objectives:
            self._optim.zero_grad()
            obj.backward(retain_graph=True)
            # On aplatit tous les gradients en un seul vecteur
            grad_vec = [p.grad.data.clone().flatten() for p in params if p.grad is not None]
            grads_per_task.append(torch.cat(grad_vec))

        # Mélange aléatoire comme dans l'implémentation originale
        random.shuffle(grads_per_task)

        # 2. Projection des gradients (Chirurgie)
        num_tasks = len(grads_per_task)
        final_grads = [g.clone() for g in grads_per_task]

        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j: continue

                dot_product = torch.dot(final_grads[i], grads_per_task[j])
                if dot_product < 0:
                    # Projection : g_i = g_i - (g_i . g_j / ||g_j||^2) * g_j
                    norm_sq = torch.dot(grads_per_task[j], grads_per_task[j])
                    final_grads[i] -= (dot_product / norm_sq) * grads_per_task[j]

        # 3. Sommer les gradients projetés et les réinjecter dans les paramètres
        merged_grad = torch.stack(final_grads).sum(dim=0)

        self._optim.zero_grad()
        offset = 0
        for p in params:
            if p.requires_grad:
                numel = p.numel()
                p.grad = merged_grad[offset:offset + numel].view_as(p).clone()
                offset += numel


class EffectiveRankMixin:

    @staticmethod
    def _safe_normalize(s, eps=1e-12):
        s = torch.clamp(s, min=eps)
        return s / s.sum()

    @staticmethod
    def effective_rank_from_singular_values(s, eps=1e-12):
        p = EffectiveRankMixin._safe_normalize(s, eps)
        entropy = -(p * torch.log(p)).sum()
        return torch.exp(entropy)

    @staticmethod
    def gram_matrix(W):
        if W.ndim > 2:
            W = W.flatten(start_dim=1)
        return W @ W.T

    @staticmethod
    def _is_weight(name, param):
        return "bias" not in name and param.ndim >= 2

    @staticmethod
    def compute_effective_ranks(model, use_gram=False, normalized=True):

        results = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not EffectiveRankMixin._is_weight(name, param):
                continue

            W = param.detach()

            if W.ndim > 2:
                W = W.flatten(start_dim=1)

            if use_gram:
                M = EffectiveRankMixin.gram_matrix(W)
            else:
                M = W

            s = torch.linalg.svdvals(M)
            erank = EffectiveRankMixin.effective_rank_from_singular_values(s)

            if normalized:
                max_rank = M.shape[0] if use_gram else min(M.shape)
                erank = erank / max_rank

            results[name] = {
                "erank": erank.item(),
                "max_rank": M.shape[0] if use_gram else min(M.shape),
            }

        return results

class GradNormTrackingMixin:
    """
    Suivi de la norme L2 des gradients couche par couche.

    Usage dans un trainer:
        - Appeler self._reset_grad_stats() en début d'époque
        - Appeler self._track_grad_norms(model) juste après loss.backward()
          (et avant optimizer.step() pour avoir les vrais gradients)
        - Appeler self._flush_grad_epoch() en fin d'époque pour agréger
          la moyenne des normes dans self.stats.grads

    self.stats.grads est une list[dict[str, dict]] de même structure
    que self.stats.ranks, ce qui permet de les plotter avec la même logique.
    """

    # ── initialisation par époque ─────────────────────────────────────────

    def _reset_grad_stats(self):
        """Réinitialise l'accumulateur de normes pour une nouvelle époque."""
        self._grad_accum: dict[str, list[float]] = {}
        self._grad_steps: int = 0

    # ── capture après chaque backward ────────────────────────────────────

    def _track_grad_norms(self, model: "nn.Module"):
        """
        Capture la norme L2 du gradient de chaque paramètre éligible
        (même filtre que EffectiveRankMixin : poids non-bias, ndim >= 2).

        À appeler juste après loss.backward(), avant optimizer.step().
        """
        self._grad_steps += 1

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or param.ndim < 2:
                continue
            if param.grad is None:
                continue

            norm = param.grad.detach().norm(2).item()

            if name not in self._grad_accum:
                self._grad_accum[name] = []
            self._grad_accum[name].append(norm)

    # ── agrégation fin d'époque ───────────────────────────────────────────

    def _flush_grad_epoch(self):
        """
        Calcule la norme moyenne sur les steps de l'époque et
        l'enregistre dans self.stats.grads (même format que stats.ranks).

        Format : list[ dict[layer_name -> {"grad_norm": float}] ]
        """
        if not hasattr(self, "_grad_accum") or self._grad_steps == 0:
            return

        epoch_entry = {
            name: {"grad_norm": sum(vals) / len(vals)}
            for name, vals in self._grad_accum.items()
        }
        self.stats.grads.append(epoch_entry)
        self._reset_grad_stats()



class ActivationTrackingMixin:
    """
    Mixin de suivi des activations avec contrôle explicite start/stop.

    Workflow dans baseTrainer :
    ────────────────────────────
        # Avant la boucle :
        self._register_activation_hooks(model)

        # En début de chaque epoch (UNE FOIS) :
        self._new_activation_epoch()

        # Dans __trainingPhase, autour de chaque forward :
        self._start_recording("train")
        preds = model(bX)
        self._stop_recording()
        # en fin de __trainingPhase :
        self._flush_activation_epoch()

        # Dans _validation_phase, autour de chaque forward :
        self._start_recording("val")
        preds = model(bX)
        self._stop_recording()
        # en fin de _validation_phase :
        self._flush_activation_epoch()

        # Après la boucle :
        self._remove_activation_hooks()

    Format de self.stats.activations :
        [
            {  # epoch 0
                "train": {"backbone": {"mean": …}, "resnet.blocks.0.act1": {…}, …},
                "val":   {…}
            },
            …
        ]

    Statistiques par couche :
        mean, std, frac_neg, frac_dead, norm
    """

    _DEAD_THRESHOLD = 1e-2

    def _register_activation_hooks(self, model: nn.Module):
        self._activation_hooks       = []
        self._activation_buffer      = defaultdict(list)
        self._activation_layer_names = []
        self._recording              = False
        self._current_phase          = "train"
        self._activation_epoch_idx   = None

        _TRACKED = (
            nn.Linear,
            nn.Conv1d, nn.Conv2d,
            nn.SiLU, nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid,
        )

        for name, module in model.named_modules():
            if isinstance(module, _TRACKED):
                self._activation_layer_names.append(name)
                hook = module.register_forward_hook(self._make_activation_hook(name))
                self._activation_hooks.append(hook)

    # ── gestion d'epoch ───────────────────────────────────────────────────────

    def _new_activation_epoch(self):
        """
        Crée une nouvelle entrée d'epoch dans stats.activations.
        À appeler UNE FOIS en début d'epoch, avant les phases train et val.
        """
        self.stats.activations.append({})
        self._activation_epoch_idx = len(self.stats.activations) - 1

    # ── contrôle d'enregistrement ─────────────────────────────────────────────

    def _start_recording(self, phase: str = "train"):
        self._current_phase = phase
        self._recording     = True

    def _stop_recording(self):
        self._recording = False

    # ── hook interne ──────────────────────────────────────────────────────────

    def _make_activation_hook(self, name: str):
        def hook(module, input, output):
            if not getattr(self, "_recording", False):
                return
            with torch.no_grad():
                x = output.detach().float()
                self._activation_buffer[name].append({
                    "mean":      x.mean().item(),
                    "std":       x.std().item(),
                    "frac_neg":  (x < 0).float().mean().item(),
                    "frac_dead": (x.abs() < self._DEAD_THRESHOLD).float().mean().item(),
                    "norm":      x.norm(2).item() / (x.numel() ** 0.5),
                })
        return hook

    # ── flush en fin de phase ─────────────────────────────────────────────────

    def _flush_activation_epoch(self):
        """
        Moyenne le buffer sur tous les batchs de la phase courante et écrit
        dans stats.activations[epoch_idx][phase]. Vide le buffer ensuite.
        """
        if not self._activation_buffer:
            return

        phase = getattr(self, "_current_phase", "train")

        epoch_stats = {}
        for name, batch_list in self._activation_buffer.items():
            keys = batch_list[0].keys()
            epoch_stats[name] = {
                k: sum(b[k] for b in batch_list) / len(batch_list)
                for k in keys
            }

        # Sécurité si _new_activation_epoch n'a pas été appelé
        if self._activation_epoch_idx is None:
            self.stats.activations.append({})
            self._activation_epoch_idx = len(self.stats.activations) - 1

        self.stats.activations[self._activation_epoch_idx][phase] = epoch_stats
        self._activation_buffer.clear()

    # ── nettoyage ─────────────────────────────────────────────────────────────

    def _remove_activation_hooks(self):
        for hook in getattr(self, "_activation_hooks", []):
            hook.remove()
        self._activation_hooks = []


class PerOutputGradMixin:
    """
    Suivi de la norme L2 des gradients par dimension de sortie (dL_i/dθ).

    Contrairement à GradNormTrackingMixin qui trace ‖dL/dθ‖ (loss agrégée),
    ce mixin calcule, pour chaque sortie i, la norme ‖dL_i/dθ‖ via
    torch.autograd.grad sur une loss non-réduite — sans toucher au graphe
    computationnel du backward principal.

    Usage dans un trainer:
        - Passer trackPerOutputGrad=True au constructeur
        - Appeler self._track_per_output_grads(model, preds, bY) AVANT
          loss.backward() (le graphe doit encore être intact)
        - Appeler self._flush_per_output_epoch() en fin d'époque

    self.stats.perOutputGrads est une list[dict[output_idx -> dict[layer -> float]]]
    """

    # ── reset début d'époque ──────────────────────────────────────────────

    def _reset_per_output_stats(self):
        """
        _accum : dict[output_idx -> dict[layer_name -> list[float]]]
        """
        self._po_accum: dict[int, dict[str, list[float]]] = {}
        self._po_steps: int = 0

    # ── capture par batch ─────────────────────────────────────────────────

    def _track_per_output_grads(
            self,
            model: "nn.Module",
            preds: "torch.Tensor",
            targets: "torch.Tensor",
    ):
        if preds.ndim == 1:
            preds = preds.unsqueeze(1)
            targets = targets.unsqueeze(1)

        self._po_steps += 1
        n_out = preds.shape[1]

        eligible = {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad and "bias" not in name and param.ndim >= 2
        }
        param_tuple = tuple(eligible.values())

        # (n_out,) — une loss scalaire par sortie
        loss_vec = nn.functional.mse_loss(
            preds, targets, reduction="none"
        ).mean(dim=0)

        # grad_outputs : n_out vecteurs identité empilés → (n_out, n_out)
        # autograd traite ça comme n_out appels backward batchés
        eye = torch.eye(n_out, device=preds.device, dtype=preds.dtype)

        jac = torch.autograd.grad(
            outputs=loss_vec,
            inputs=param_tuple,
            grad_outputs=eye,  # ← (n_out, n_out) : une ligne par sortie
            retain_graph=True,  # ← graphe intact pour loss.backward()
            create_graph=False,
            is_grads_batched=True,  # ← dit à autograd que grad_outputs est batché
        )
        # jac : tuple[ (n_out, *param.shape) ] — même format qu'avant

        for (name, _), jac_p in zip(eligible.items(), jac):
            norms = jac_p.flatten(start_dim=1).norm(dim=1)  # (n_out,)
            for i, val in enumerate(norms.tolist()):
                self._po_accum.setdefault(i, {}).setdefault(name, []).append(val)

    # ── agrégation fin d'époque ───────────────────────────────────────────

    def _flush_per_output_epoch(self):
        """
        Moyenne les normes sur les steps de l'époque et pousse dans
        self.stats.perOutputGrads.

        Format : list[ dict[int -> dict[layer_name -> {"grad_norm": float}]] ]
        """
        if not hasattr(self, "_po_accum") or self._po_steps == 0:
            return

        epoch_entry = {
            i: {
                name: {"grad_norm": sum(vals) / len(vals)}
                for name, vals in layers.items()
            }
            for i, layers in self._po_accum.items()
        }
        self.stats.perOutputGrads.append(epoch_entry)
        self._reset_per_output_stats()
from POLENE_GPU.NN.models._wrapper import _wrapper

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComposedSourceIntegratorNet(_wrapper):
    """
    Compose deux réseaux indépendants pour l'entraînement en deux temps :

        x (TPY) ──► [sourceNet] ──► omega_hat
         │                              │
         └──────► concat(x, omega_hat) ─┤
                                         ▼
                                 [integratorNet] ──► output (TPY(t+dt) scalé)

    - sourceNet      : prédit les termes sources omega. Généralement
                        pré-entraîné séparément puis gelé (freeze_source=True).
    - integratorNet  : prend concat(TPY, omega_hat) et prédit TPY(t+dt).
                        Entraîné from scratch (freeze_integrator=False).

    sourceNet et integratorNet peuvent être n'importe quel nn.Module. S'ils
    exposent torchForward (cas des sous-classes de _wrapper), elle est
    utilisée directement pour éviter l'ambiguïté de forward(serie, tensorInput)
    défini par _wrapper. Sinon net(x) est appelé normalement.

    predictState doit rester True : integratorNet produit directement l'état
    TPY reconstruit (pas un résidu omega), donc le pipeline standard de
    baseTrainer (`if model.predictState: preds = model(bX, tensorInput=True)`)
    s'applique tel quel, sans toucher au trainer.
    """

    def __init__(
        self,
        sourceNet,
        integratorNet,
        freeze_source=True,
        freeze_integrator=False,
        predictState=True,
        chemScaler=None,
        chemStdScaler=None,
    ):
        nn.Module.__init__(self)
        _wrapper.__init__(self, predictState, chemScaler, chemStdScaler)

        if not predictState:
            raise ValueError(
                "ComposedSourceIntegratorNet ne supporte que predictState=True : "
                "la sortie de integratorNet est déjà l'état reconstruit, pas un résidu."
            )

        self.sourceNet = sourceNet
        self.integratorNet = integratorNet

        self.freeze_source = False
        self.freeze_integrator = False
        self.enableFineTuning(freeze_source=freeze_source, freeze_integrator=freeze_integrator)

    # ---- gel / dégel ------------------------------------------------------

    @staticmethod
    def _setRequiresGrad(module, flag):
        for p in module.parameters():
            p.requires_grad = flag

    def enableFineTuning(self, freeze_source=False, freeze_integrator=False):
        """
        Gèle (True) ou dégèle (False) chaque sous-réseau : coupe/rétablit
        requires_grad ET bascule train()/eval() en conséquence, pour que les
        couches sensibles au mode (Dropout, BatchNorm...) se comportent
        correctement. L'état est ensuite préservé automatiquement par la
        surcharge de train() ci-dessous, même quand le trainer appelle
        model.train() sur l'ensemble à chaque époque.

        Peut être rappelée entre deux entraînements successifs pour changer
        la stratégie de gel (ex: dégeler sourceNet pour du fine-tuning joint
        après une première phase) — _build_optimizer reconstruira alors un
        optimizer qui inclut les paramètres nouvellement entraînables.
        """
        self.freeze_source = freeze_source
        self.freeze_integrator = freeze_integrator

        self._setRequiresGrad(self.sourceNet, not freeze_source)
        self._setRequiresGrad(self.integratorNet, not freeze_integrator)

        self.sourceNet.train(not freeze_source)
        self.integratorNet.train(not freeze_integrator)

    def disableFineTuning(self):
        """Dégèle tout (les deux sous-réseaux redeviennent entraînables)."""
        self.enableFineTuning(freeze_source=False, freeze_integrator=False)

    def train(self, mode=True):
        super().train(mode)
        self.sourceNet.train(mode and not self.freeze_source)
        self.integratorNet.train(mode and not self.freeze_integrator)
        return self

    # ---- forward ------------------------------------------------------------

    @staticmethod
    def _forwardSubnet(net, x):
        if hasattr(net, "torchForward"):
            return net.torchForward(x)
        return net(x)

    def torchForward(self, x):
        if self.freeze_source:
            with torch.no_grad():
                omega_hat = self._forwardSubnet(self.sourceNet, x)
        else:
            omega_hat = self._forwardSubnet(self.sourceNet, x)

        fused = torch.cat([x, omega_hat], dim=-1)
        return self._forwardSubnet(self.integratorNet, fused)

    # ---- config / persistance ------------------------------------------------

    def _get_config(self):
        # sourceNet/integratorNet sont des nn.Module arbitraires, non
        # reconstructibles depuis un simple dict scalaire (contrairement à
        # ce que suppose _wrapper._from_config pour un modèle "feuille").
        # On ne sauvegarde donc que l'état de gel ; la reconstruction de
        # l'architecture est à la charge de l'appelant via load()/
        # fromPretrainedSource() ci-dessous.
        return {
            "freeze_source": self.freeze_source,
            "freeze_integrator": self.freeze_integrator,
        }

    @classmethod
    def _from_config(cls, config):
        raise NotImplementedError(
            "ComposedSourceIntegratorNet ne peut pas être reconstruit depuis sa "
            "seule config : instancie sourceNet/integratorNet toi-même et utilise "
            "load(path, sourceNet, integratorNet) ou fromPretrainedSource(...)."
        )

    @classmethod
    def load(cls, path, sourceNet, integratorNet):
        """
        Signature différente de _wrapper.load(path) à dessein : sourceNet et
        integratorNet doivent être instanciés par l'appelant (même
        architecture qu'au moment du save), puis load_state_dict restaure
        les poids des deux sous-réseaux.
        """
        checkpoint = torch.load(path, weights_only=False)
        cfg = checkpoint["config"]
        model = cls(
            sourceNet, integratorNet,
            freeze_source=cfg["freeze_source"],
            freeze_integrator=cfg["freeze_integrator"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.inputScaler  = pickle.loads(checkpoint["scalers"]["input"])
        model.outputScaler = pickle.loads(checkpoint["scalers"]["output"])
        return model

    @classmethod
    def fromPretrainedSource(
        cls, sourceNetClass, sourceNetPath, integratorNet,
        freeze_source=True, freeze_integrator=False,
    ):
        """
        Raccourci pour ton workflow : sourceNet a déjà été entraîné et
        sauvegardé séparément (typiquement un _wrapper avec predictState=False
        prédisant omega), integratorNet est neuf (poids aléatoires).

        sourceNetClass doit implémenter _from_config/_get_config (interface
        _wrapper standard) pour que .load() fonctionne.
        """
        sourceNet = sourceNetClass.load(sourceNetPath)
        return cls(sourceNet, integratorNet, freeze_source=freeze_source, freeze_integrator=freeze_integrator)
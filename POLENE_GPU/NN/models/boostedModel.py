"""
boostedModel.py  →  à placer dans POLENE_GPU/NN/models/
────────────────────────────────────────────────────────
Wrapper "boosting" : un réseau de base (omega, éventuellement cutté) + un réseau
correcteur entraîné sur l'erreur du premier dans l'espace BCT non cutté.

Interface identique à MixtureOfExperts :
  - prend des états PHYSIQUES en entrée, rend des états PHYSIQUES en sortie
  - predictState=True, inputScaler/outputScaler = identité
    => compatible avec tous les validateurs existants

    y1   = base(x)                              (état physique, via omega)
    z    = BCT_full(clamp(y1)) + booster(x)     (correction dans l'espace BCT non cutté)
    y    = BCT_full^-1(z)

Save / load : hérités de _wrapper (tout est sérialisé dans _get_config).

    boosted = BoostedModel(model, error_corr_model, BCTfull)
    boosted.to('cpu'); boosted.save('checkpoints', 'boosted.pt')
    boosted = BoostedModel.load('checkpoints/boosted.pt')
"""

import importlib
import pickle

import torch
import torch.nn as nn

from POLENE_GPU.NN.models._wrapper import _wrapper
from POLENE_GPU.preProcessing.scaler.identityScaler import identityScaler


def _pack(m):
    """Sérialise un réseau + tout ce dont _wrapper.load n'a pas connaissance."""
    return {
        "cls_module":    type(m).__module__,
        "cls_name":      type(m).__qualname__,
        "config":        m._get_config(),
        "scalers": {
            "input":  pickle.dumps(m.inputScaler),
            "output": pickle.dumps(m.outputScaler),
        },
        # ResNet._get_config() ne contient pas ces champs : sans eux, un réseau
        # omega rechargé perdrait predictState=False et son chemScaler.
        "predictState":  m.predictState,
        "chemScaler":    pickle.dumps(m.chemScaler),
        "chemStdScaler": pickle.dumps(m.chemStdScaler),
    }


def _unpack(payload):
    cls = importlib.import_module(payload["cls_module"])
    for part in payload["cls_name"].split("."):
        cls = getattr(cls, part)
    m = cls._from_config(payload["config"])
    m.inputScaler = pickle.loads(payload["scalers"]["input"])
    m.outputScaler = pickle.loads(payload["scalers"]["output"])
    m.predictState = payload["predictState"]
    m.chemScaler = pickle.loads(payload["chemScaler"])
    m.chemStdScaler = pickle.loads(payload["chemStdScaler"])
    return m


class BoostedModel(_wrapper):
    """
    Parameters
    ----------
    base      : _wrapper   réseau initial (omega : predictState=False, ou état : predictState=True)
    booster   : _wrapper   réseau correcteur, sortie = erreur BCT normalisée
                           (booster.inputScaler = scaler d'entrée, booster.outputScaler = scaler d'erreur)
    bctScaler : scaler BCT NON cutté (threshold=0), celui dans lequel l'erreur a été définie
    clampMin  : plancher physique appliqué à l'état du réseau de base avant passage en BCT
    """

    def __init__(self, base, booster, bctScaler, clampMin=1e-30):
        nn.Module.__init__(self)
        _wrapper.__init__(self)

        self.base = base
        self.booster = booster
        self.bctScaler = bctScaler
        self.clampMin = clampMin

        # Comme MixtureOfExperts : le wrapper reçoit/rend du physique, les scalers
        # sont gérés en interne par chaque sous-réseau.
        self.inputScaler = identityScaler()
        self.outputScaler = identityScaler()

    # ──────────────────────────────────────────────────────────────────────
    def baseState(self, x: torch.Tensor) -> torch.Tensor:
        """État physique prédit par le réseau de base seul (non clampé)."""
        b = self.base
        out = b(b.inputScaler.transformTorch(x), tensorInput=True)
        if b.predictState:
            return b.outputScaler.inverse_transformTorch(out)
        return b.torchOutput2State(x, out)

    def boostCorrection(self, x: torch.Tensor) -> torch.Tensor:
        """Correction prédite par le booster, en espace BCT non cutté (dé-normalisée)."""
        e_scaled = self.booster(self.booster.inputScaler.transformTorch(x), tensorInput=True)
        return self.booster.outputScaler.inverse_transformTorch(e_scaled)

    def torchForward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = torch.clamp(self.baseState(x), min=self.clampMin)
        z = self.bctScaler.transformTorch(y1) + self.boostCorrection(x)

        # BCT^-1 n'est défini que pour 1 + lambda*z >= 0 : on borne pour éviter les NaN
        lam = getattr(self.bctScaler, "lambda_", None)
        if lam:
            z = torch.clamp(z, min=-1.0 / lam)
        return self.bctScaler.inverse_transformTorch(z)

    # ──────────────────────────────────────────────────────────────────────
    def _get_config(self):
        return {
            "base":      _pack(self.base),
            "booster":   _pack(self.booster),
            "bctScaler": pickle.dumps(self.bctScaler),
            "clampMin":  self.clampMin,
        }

    @classmethod
    def _from_config(cls, config):
        return cls(
            _unpack(config["base"]),
            _unpack(config["booster"]),
            pickle.loads(config["bctScaler"]),
            config["clampMin"],
        )

    def __repr__(self):
        return (f"BoostedModel(\n  base    = {type(self.base).__name__}\n"
                f"  booster = {type(self.booster).__name__}\n)")
from POLENE_GPU.preProcessing.scaler._BaseTransform import _BaseTransform
from POLENE_GPU.dataHandler.serie import serie

import torch


class boxCoxScaler(_BaseTransform):
    def __init__(self, threshold=1e-30, lambda_=0.1):
        if lambda_ == 0:
            raise ValueError(
                "boxCoxScaler does not support lambda=0 (this degenerates to the "
                "log transform); use logScaler instead."
            )

        self.threshold = threshold
        self.lambda_ = lambda_

        self._species_mask = None
        self._col_names = None
        self._device = None
        self._dtype = None

    def _build_species_mask(self, col_names, device):
        return torch.tensor(
            [name not in self.NON_SPECIES_COLS for name in col_names],
            dtype=torch.bool,
            device=device,
        )

    def _sync(self, x):
        self._species_mask = self._species_mask.to(x.storingDevice)
        self._device = x.storingDevice
        self._dtype = x.storingDtype

    def _syncTorch(self, x):
        # .to() est un no-op (pas de copie, pas de sync) si le masque est déjà sur le bon device
        self._species_mask = self._species_mask.to(x.device)
        self._device = x.device
        self._dtype = x.dtype

    def _threshold_tensor(self, device, dtype):
        return torch.tensor(self.threshold, device=device, dtype=dtype)

    def _clip(self, data):
        # Version sans indexation par masque booléen (pas de nonzero -> pas de sync GPU)
        return torch.where(
            self._species_mask,
            torch.clamp(data, min=self.threshold),
            data,
        )


############################################
    #TODO: try to understand the impact of the threshold and of the clamping on grandient propagation and on autoregressive use
    # Les deux fonctions ci-dessous n'utilisent AUCUNE indexation par masque booléen
    # (data[:, mask] / out[:, mask] = ... déclenchent un nonzero => synchronisation CPU<->GPU)
    # et aucune création de tenseur à chaque appel (le seuil est un scalaire Python).
    #
    # Double torch.where : les colonnes non-espèces (T, P, ...) sont remplacées par 1.0
    # AVANT le pow, pour que la branche non sélectionnée ne produise jamais d'inf/NaN
    # (sinon 0 * inf = NaN dans le gradient, p.ex. (lam*P + 1)**10 déborde en float32).
    def _forward(self, data):
        lam = self.lambda_
        mask = self._species_mask  # (F,), broadcast sur (N, F)

        col = torch.where(mask, data, torch.ones_like(data))
        safe_col = torch.clamp(col, min=self.threshold)  # jamais <= 0 si threshold > 0
        bct = (safe_col.pow(lam) - 1.0) / lam
        return torch.where(mask, bct, data)

    def _backward(self, data):
        lam = self.lambda_
        mask = self._species_mask

        base = lam * data + 1.0
        base = torch.where(mask, base, torch.ones_like(base))
        base = torch.clamp(base, min=self.threshold)  # évite pow() sur base négative/nulle #TODO: saise une erreur ou warning si on clamp réellement, ça veut dire qu'on sort du domaine de la trasformation box cox
        return torch.where(mask, base.pow(1.0 / lam), data)
#########################################


    def fit(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")

        col_names = list(x.colName)
        self._col_names = col_names
        self._species_mask = self._build_species_mask(col_names, x.storingDevice)
        self._device = x.storingDevice
        self._dtype = x.storingDtype
        return self

    def transform(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        return serie(
            self._forward(x.data),
            x.colName,
            storingDevice=x.storingDevice,
            storingDtype=x.storingDtype,
        )

    def transformTorch(self, x):
        self._syncTorch(x)
        return (self._forward(x))

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        return serie(
            self._backward(x.data),
            x.colName,
            storingDevice=x.storingDevice,
            storingDtype=x.storingDtype,
        )

    def inverse_transformTorch(self, x):
        self._syncTorch(x)
        return (self._backward(x))

    def J(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        N, F = x.data.shape
        J = torch.ones(N, F, device=x.storingDevice, dtype=x.storingDtype)

        t = self._threshold_tensor(x.storingDevice, x.storingDtype)
        lam = self.lambda_
        col = x.data[:, self._species_mask]

        above = col > t
        safe_col = torch.where(above, col, torch.ones_like(col))
        # dF/dx = x ** (lambda - 1)
        J[:, self._species_mask] = torch.where(
            above, safe_col.pow(lam - 1.0), torch.zeros_like(col)
        )

        return J.to(x.storingDevice).to(x.storingDtype)

    def J_1(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        N, F = x.data.shape
        J_inv = torch.ones(N, F, device=x.storingDevice, dtype=x.storingDtype)

        lam = self.lambda_
        col = x.data[:, self._species_mask]

        # dx/dy = (lambda * y + 1) ** (1/lambda - 1)
        base = lam * col + 1.0
        J_inv[:, self._species_mask] = base.pow(1.0 / lam - 1.0)

        return J_inv.to(x.storingDevice).to(x.storingDtype)

    def __repr__(self) -> str:
        return f"boxCoxScaler(threshold={self.threshold}, lambda_={self.lambda_})"
from POLENE_GPU.preProcessing.scaler._BaseTransform import _BaseTransform
from POLENE_GPU.dataHandler.serie import serie

import torch


class logScaler(_BaseTransform):
    def __init__(self, threshold):
        self.threshold = threshold

        self._species_mask = None
        self._col_names = None
        self._device = None
        self._dtype  = None

    def _build_species_mask(self, col_names, device):
        return torch.tensor(
            [name not in self.NON_SPECIES_COLS for name in col_names],
            dtype=torch.bool,
            device=device,
        )

    def _sync(self, x):
        self._species_mask = self._species_mask.to(x.storingDevice)
        self._device = x.storingDevice
        self._dtype  = x.storingDtype

    def _syncTorch(self, x):
        self._species_mask = self._species_mask.to(x.device)
        self._device = x.device
        self._dtype  = x.dtype

    def _threshold_tensor(self, device, dtype):
        return torch.tensor(self.threshold, device=device, dtype=dtype)

    def _clip(self, data):
        t = self._threshold_tensor(data.device, data.dtype)
        out = data.clone()
        out[:, self._species_mask] = torch.where(
            data[:, self._species_mask] <= t,
            t,
            data[:, self._species_mask],
        )
        return out

    def _forward(self, data):
        out = data.clone()
        t = self._threshold_tensor(data.device, data.dtype)

        col = data[:, self._species_mask]
        safe_col = torch.where(col > t, col, t)  # jamais <= 0 si threshold > 0
        out[:, self._species_mask] = safe_col.log()
        return out

    def _backward(self, data):
        out = data.clone()
        out[:, self._species_mask] = data[:, self._species_mask].exp()
        return out

    def fit(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")

        col_names = list(x.colName)
        self._col_names    = col_names
        self._species_mask = self._build_species_mask(col_names, x.storingDevice)
        self._device = x.storingDevice
        self._dtype  = x.storingDtype
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
        return(self._forward(x))

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
        return(self._backward(x))

    def J(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        N, F = x.data.shape
        J = torch.ones(N, F, device=x.storingDevice, dtype=x.storingDtype)

        t = self._threshold_tensor(x.storingDevice, x.storingDtype)
        col = x.data[:, self._species_mask]

        above = col > t
        safe_col = torch.where(above, col, torch.ones_like(col))
        J[:, self._species_mask] = torch.where(above, 1.0 / safe_col, torch.zeros_like(col))

        return J.to(x.storingDevice).to(x.storingDtype)

    def J_1(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        N, F = x.data.shape
        J_inv = torch.ones(N, F, device=x.storingDevice, dtype=x.storingDtype)
        J_inv[:, self._species_mask] = x.data[:, self._species_mask].exp()

        return J_inv.to(x.storingDevice).to(x.storingDtype)

    def __repr__(self) -> str:
        return f"logScaler(threshold={self.threshold})"
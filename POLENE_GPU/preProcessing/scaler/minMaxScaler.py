from POLENE_GPU.preProcessing.scaler._BaseTransform import _BaseTransform
from POLENE_GPU.dataHandler.serie import serie

import torch


class minMaxScaler(_BaseTransform):
    """
    Met à l'échelle chaque colonne vers `feature_range` (par défaut [0, 1]) :

        x_scaled = (x - min) / (max - min) * (hi - lo) + lo

    Les colonnes constantes (max - min nul ou négligeable) sont envoyées sur `lo`
    à la transformation, et reviennent à `min` à l'inverse. Leur jacobien est 0
    (même convention que stdScaler).
    """

    NULL_RELATIVE_RANGE = 1e-12

    def __init__(self, feature_range=(-1.0, 1.0), rangeCutoff=1e-28):
        lo, hi = feature_range
        if not hi > lo:
            raise ValueError("feature_range must be (lo, hi) with hi > lo")

        self.feature_range = (float(lo), float(hi))
        self.rangeCutoff = rangeCutoff

        self.min = None
        self.max = None
        self.range = None
        self.is_constant = None
        self._input_ndim = None

    def __checkdim(self, x):
        if x.dim() not in (2, 3):
            raise ValueError("Input must be 2D (N, F) or 3D (N, H, W)")

    def __checkdim_serie(self, x):
        self.__checkdim(x.data)

    def __changeDeviceTorch(self, x):
        self.min         = self.min.to(x.device).to(x.dtype)
        self.max         = self.max.to(x.device).to(x.dtype)
        self.range       = self.range.to(x.device).to(x.dtype)
        self.is_constant = self.is_constant.to(x.device)

    def __changeDevice(self, x):
        self.__changeDeviceTorch(x.data)

    def __check_fitted(self):
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler not fitted")

    def __check_not_matrix_for_jacobian(self):
        if self._input_ndim == 3:
            raise RuntimeError(
                "J and J_1 are not defined when the scaler was fitted on 3D inputs (N, H, W). "
                "Flatten your input to (N, H*W) before using J / J_1."
            )

    def fit(self, x):
        self.__checkdim_serie(x)
        self._input_ndim = x.data.dim()

        self.min   = x.data.min(dim=0).values
        self.max   = x.data.max(dim=0).values
        self.range = self.max - self.min

        # étendue relative : range / max(|min|, |max|), calculée seulement là où l'échelle est non nulle
        scale = torch.maximum(self.min.abs(), self.max.abs())
        zero_range = (self.range == 0)
        zero_scale = (scale == 0)
        safe_rel = torch.where(
            zero_range | zero_scale,
            torch.zeros_like(self.range),
            self.range / scale,
        )

        self.is_constant = zero_range | (safe_rel < self.NULL_RELATIVE_RANGE) | (self.range < self.rangeCutoff)

        return self

    # ------------------------------------------------------------------
    # transform / inverse_transform  →  délèguent aux versions Torch
    # ------------------------------------------------------------------

    def transform(self, x):
        self.__checkdim_serie(x)
        self.__changeDevice(x)
        self.__check_fitted()

        scaled = self.transformTorch(x.data)
        return serie(scaled, x.colName, storingDevice=x.storingDevice, storingDtype=x.storingDtype)

    def transformTorch(self, x):
        self.__checkdim(x)
        self.__check_fitted()
        self.__changeDeviceTorch(x)

        lo, hi = self.feature_range
        safe_range = torch.where(self.is_constant, torch.ones_like(self.range), self.range)
        scaled = (x - self.min) / safe_range * (hi - lo) + lo
        scaled = torch.where(self.is_constant, torch.full_like(scaled, lo), scaled)
        return scaled

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        self.__checkdim_serie(x)
        self.__changeDevice(x)
        self.__check_fitted()

        unscaled = self.inverse_transformTorch(x.data)
        return serie(unscaled, x.colName, storingDevice=x.storingDevice, storingDtype=x.storingDtype)

    def inverse_transformTorch(self, x):
        self.__checkdim(x)
        self.__check_fitted()
        self.__changeDeviceTorch(x)

        lo, hi = self.feature_range
        safe_range = torch.where(self.is_constant, torch.ones_like(self.range), self.range)
        unscaled = (x - lo) / (hi - lo) * safe_range + self.min
        unscaled = torch.where(self.is_constant, self.min.expand_as(unscaled), unscaled)
        return unscaled

    # ------------------------------------------------------------------
    # Jacobiens — définis uniquement pour les entrées 2D
    # ------------------------------------------------------------------

    def J(self, x):
        self.__checkdim_serie(x)
        self.__check_not_matrix_for_jacobian()
        self.__check_fitted()

        lo, hi = self.feature_range

        J_diag = torch.zeros_like(self.range)
        mask = ~self.is_constant
        J_diag[mask] = (hi - lo) / self.range[mask]

        return J_diag.unsqueeze(0).expand(x.nSamples, -1).to(x.storingDevice).to(x.storingDtype)

    def J_1(self, x):
        self.__checkdim_serie(x)
        self.__check_not_matrix_for_jacobian()
        self.__check_fitted()

        lo, hi = self.feature_range

        J_diag = torch.zeros_like(self.range)
        mask = ~self.is_constant
        J_diag[mask] = self.range[mask] / (hi - lo)

        return J_diag.unsqueeze(0).expand(x.nSamples, -1).to(x.storingDevice).to(x.storingDtype)
from POLENE_GPU.preProcessing.scaler._BaseTransform import _BaseTransform
from POLENE_GPU.dataHandler.serie import serie

import torch

class stdScaler(_BaseTransform):

    NULL_RELATIVE_VARIANCE = 1e-12

    def __init__(self, varianceCutoff=1e-28):  # TODO: faire un scaler "filterScaler" qui filtre les valeurs minuscules. comme ça ou pourait par exemple le mettre avant un scaler log, c'est une forme de pretraitement qui vire des colonnes. sinon vraiement faire une fonction de pretraitement qui met à 0
        self.mean = None
        self.std = None
        self.is_constant = None
        self._input_ndim = None
        self.varianceCutoff = varianceCutoff

    def __checkdim(self, x):
        if x.dim() not in (2, 3):
            raise ValueError("Input must be 2D (N, F) or 3D (N, H, W)")

    def __checkdim_serie(self, x):
        self.__checkdim(x.data)

    def __changeDeviceTorch(self, x):
        self.std         = self.std.to(x.device).to(x.dtype)
        self.mean        = self.mean.to(x.device).to(x.dtype)
        self.is_constant = self.is_constant.to(x.device)

    def __changeDevice(self, x):
        self.__changeDeviceTorch(x.data)

    def __check_not_matrix_for_jacobian(self):
        if self._input_ndim == 3:
            raise RuntimeError(
                "J and J_1 are not defined when the scaler was fitted on 3D inputs (N, H, W). "
                "Flatten your input to (N, H*W) before using J / J_1."
            )

    def fit(self, x):
        self.__checkdim_serie(x)
        self._input_ndim = x.data.dim()

        self.mean = x.data.mean(dim=0)
        self.std  = x.data.std(dim=0)

        zero_std = (self.std == 0)
        zero_mean = (self.mean == 0)

        # cv n'est calculé que là où mean != 0 et std != 0
        safe_cv = torch.where(zero_std | zero_mean, torch.zeros_like(self.std), self.std / self.mean.abs())
        self.is_constant = zero_std | (safe_cv < self.NULL_RELATIVE_VARIANCE) | (self.std < self.varianceCutoff)

        return self

    # ------------------------------------------------------------------
    # transform / inverse_transform  →  délèguent aux versions Torch
    # ------------------------------------------------------------------

    def transform(self, x):
        self.__checkdim_serie(x)
        self.__changeDevice(x)

        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")

        scaled = self.transformTorch(x.data)
        return serie(scaled, x.colName, storingDevice=x.storingDevice, storingDtype=x.storingDtype)

    def transformTorch(self, x):
        self.__checkdim(x)
        self.__changeDeviceTorch(x)

        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")

        safe_std = torch.where(self.is_constant, torch.ones_like(self.std), self.std)
        scaled = (x - self.mean) / safe_std
        scaled = torch.where(self.is_constant, torch.zeros_like(scaled), scaled)
        return scaled

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        self.__checkdim_serie(x)
        self.__changeDevice(x)

        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")

        unscaled = self.inverse_transformTorch(x.data)
        return serie(unscaled, x.colName, storingDevice=x.storingDevice, storingDtype=x.storingDtype)

    def inverse_transformTorch(self, x):
        self.__checkdim(x)
        self.__changeDeviceTorch(x)

        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")

        safe_std = torch.where(self.is_constant, torch.ones_like(self.std), self.std)
        unscaled = x * safe_std + self.mean
        unscaled = torch.where(self.is_constant, self.mean.expand_as(unscaled), unscaled)
        return unscaled

    # ------------------------------------------------------------------
    # Jacobiens — définis uniquement pour les entrées 2D
    # ------------------------------------------------------------------

    def J(self, x):
        self.__checkdim_serie(x)
        self.__check_not_matrix_for_jacobian()

        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")

        J_diag = torch.zeros_like(self.std)
        mask = ~self.is_constant
        J_diag[mask] = 1.0 / self.std[mask]

        return J_diag.unsqueeze(0).expand(x.nSamples, -1).to(x.storingDevice).to(x.storingDtype)

    def J_1(self, x):
        self.__checkdim_serie(x)
        self.__check_not_matrix_for_jacobian()

        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")

        J_diag = torch.zeros_like(self.std)
        mask = ~self.is_constant
        J_diag[mask] = self.std[mask]

        return J_diag.unsqueeze(0).expand(x.nSamples, -1).to(x.storingDevice).to(x.storingDtype)
from POLENE_GPU.preProcessing.scaler._BaseTransform import _BaseTransform
from POLENE_GPU.dataHandler.serie import serie

import torch

class stdScaler(_BaseTransform):

    NULL_RELATIVE_VARIANCE = 1e-12  # TODO: look if the value depends on float vs double input

    def __init__(self):
        self.mean = None
        self.std = None
        self.is_constant = None

    def __checkdim(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")

    def __changeDevice(self, x):
        self.std = self.std.to(x.storingDevice).to(x.storingDtype)
        self.mean = self.mean.to(x.storingDevice).to(x.storingDtype)
        self.is_constant = self.is_constant.to(x.storingDevice)

    def __changeDeviceTorch(self, x):
        self.std = self.std.to(x.device).to(x.dtype)
        self.mean = self.mean.to(x.device).to(x.dtype)
        self.is_constant = self.is_constant.to(x.device)

    def fit(self, x):
        self.__checkdim(x)

        self.mean = x.data.mean(dim=0)
        self.std = x.data.std(dim=0)

        # coefficient de variation (stable même si mean=0 grace à la condition qui suit)
        cv = self.std / self.mean.abs()

        # mask strict des constantes
        self.is_constant = (self.std == 0) | (cv < self.NULL_RELATIVE_VARIANCE)

        return self

    def transform(self, x):
        self.__checkdim(x)
        self.__changeDevice(x)

        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")

        # normal features
        scaled = (x.data - self.mean) / self.std

        # apply mask AFTER computation (safe because we'll overwrite)
        scaled = torch.where(self.is_constant, torch.zeros_like(scaled), scaled)

        return serie(scaled, x.colName, storingDevice=x.storingDevice, storingDtype=x.storingDtype)

    def transformTorch(self, x):
        self.__checkdim(x)
        self.__changeDeviceTorch(x)

        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")

        # normal features
        scaled = (x - self.mean) / self.std

        # apply mask AFTER computation (safe because we'll overwrite)
        scaled = torch.where(self.is_constant, torch.zeros_like(scaled), scaled)

        return scaled

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        self.__checkdim(x)
        self.__changeDevice(x)

        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")

        unscaled = x.data * self.std + self.mean

        # constant features stay constant
        unscaled = torch.where(self.is_constant,self.mean,unscaled)

        return serie(unscaled, x.colName, storingDevice=x.storingDevice, storingDtype=x.storingDtype)

    def inverse_transformTorch(self, x):
        self.__checkdim(x)
        self.__changeDeviceTorch(x)

        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")

        unscaled = x * self.std + self.mean

        # constant features stay constant
        unscaled = torch.where(self.is_constant,self.mean,unscaled)

        return unscaled

    def J(self, x):
        self.__checkdim(x)

        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")

        # diag entries
        J_diag = torch.zeros_like(self.std)

        # only non-constant features
        mask = ~self.is_constant
        J_diag[mask] = 1.0 / self.std[mask]

        # expand to batch
        # shape: (N, F)
        return J_diag.unsqueeze(0).expand(x.nSamples, -1).to(x.storingDevice).to(x.storingDtype)

    def J_1(self, x):
        self.__checkdim(x)

        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fitted")

        J_diag = torch.zeros_like(self.std)

        mask = ~self.is_constant
        J_diag[mask] = self.std[mask]

        return J_diag.unsqueeze(0).expand(x.nSamples, -1).to(x.storingDevice).to(x.storingDtype)
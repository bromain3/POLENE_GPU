# define nSamples  :  OK
# defnine a concat that works with lists of lists of series
# define the serie with
    # initializing from a list of arraylike  :  OK
    # suited for storing 1D, 2D and 3D arrays (list of 1D features or list of 1 Jacobian)  :  OK
    # implementing serie['X'] notation  :  OK
    # implementing a serie[i:j] or serie[i] notation  :  OK
    # implementing a serie[torch.tensor([1,2,3]) notation  :  OK
    # TODO : implementing .mean() .abs() .pow() .item() .cpu() .detach() .numpy() .item()

import numpy as np
import torch

class serie:
    def __init__(self, data, colName, storingDevice=None, storingDtype=None):
        self.colName = colName
        self.ncols = len(colName)
        self.data = self._to_tensor(data, device=storingDevice, dtype=storingDtype)
        self.data = self._normalize_shape(self.data)
        self.storingDevice = storingDevice
        self.storingDtype = storingDtype


    def _to_tensor(self, data, device=None, dtype=None):
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=dtype) if (device or dtype) else data

        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device=device, dtype=dtype)

        return torch.tensor(data, device=device, dtype=dtype)

    def _normalize_shape(self, t):
        # scalar
        if t.dim() == 0:
            return t.unsqueeze(0).unsqueeze(1)

        # 1D case → disambiguation
        if t.dim() == 1:
            if self.ncols == 1:
                # single feature → column vector (N,1)
                return t.unsqueeze(1)

            elif self.ncols == len(t):
                # single sample → row vector (1,N)
                return t.unsqueeze(0)

            else:
                raise ValueError(f"Ambiguous shape: len(data)={len(t)} vs ncols={self.ncols}")

        return t

    @classmethod
    def _from_raw(cls, tensor, colName, storingDevice, storingDtype):
        """Constructeur rapide : skip validation, pour usage interne uniquement."""
        obj = object.__new__(cls)
        obj.colName = colName
        obj.ncols = len(colName)
        obj.data = tensor
        obj.storingDevice = storingDevice
        obj.storingDtype = storingDtype
        return obj

    @property
    def nSamples(self):
        return self.data.shape[0]

    @property
    def nFeatures(self):
        return self.ncols

    def to(self, device):
        self.data = self.data.to(device)
        self.storingDevice = self.data.device
        self.storingDtype = self.data.dtype
        return self

    def contiguous(self):
        self.data = self.data.contiguous()
        return self

    def __getitem__(self, key):
        #raise TypeError()  # if key is not of the right type
        #raise IndexError()  # if key is numerical but outside the range of allowed values (used by for loops to stop)
        #raise KeyError()  # if key is a special type and cannot be accessed
        if isinstance(key, str):
            idx = self.colName.index(key)
            # CAS 1 : plusieurs colonnes 2D
            if self.data.dim() >= 2 and self.nSamples > 1:
                return self.data[:, idx]
            # CAS 2 : une seule colonne ND
            if self.data.dim() > 2 and self.ncols == 1:
                return self.data  # shape (N, ...)
            # CAS 3 : plusieurs colones mais une seule ligne (retour scalaire)
            if self.nSamples == 1:
                return self.data[0, idx].detach().cpu().item()
            else:
                raise ValueError("shape/columns not coherent")

        elif isinstance(key, (int, slice, list, tuple, torch.Tensor, np.ndarray)):
            return serie(self.data[key], self.colName, storingDevice=self.storingDevice, storingDtype=self.storingDtype)

        else:
            raise KeyError(f"type {type(key)} not supported as serie key")


def concat(a):
    if not isinstance(a, (list, tuple)) or len(a) == 0:
        raise TypeError("Input must be a non-empty list")

    # CAS 1 : liste de series
    if all(isinstance(x, serie) for x in a):
        first = a[0]
        for s in a:
            if s.colName != first.colName:
                raise ValueError("All series must have the same columns")
        data = torch.vstack([s.data for s in a])
        return serie._from_raw(data, first.colName, first.storingDevice, first.storingDtype)

    else:
        raise TypeError("Input must be a list of series")
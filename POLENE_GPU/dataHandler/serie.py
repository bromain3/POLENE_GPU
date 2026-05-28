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
import zarr
import json
from numcodecs import Blosc
from concurrent.futures import ThreadPoolExecutor
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp

# Compresseur LZ4 : meilleur compromis vitesse/ratio pour des tenseurs numériques
# clevel=1 → compression légère mais décompression maximale (~3-5 GB/s)
# shuffle=Blosc.BITSHUFFLE → améliore ratio sur float32/float64
_COMPRESSOR = Blosc(cname='lz4', clevel=1, shuffle=Blosc.BITSHUFFLE)


# ─── workers (top-level pour être picklables) ───────────────────────────────

def _write_chunk_worker(args):
    filepath, shm_name, shape, dtype, chunk_rows, chunk_idx = args
    start = chunk_idx * chunk_rows
    end = min(start + chunk_rows, shape[0])

    shm = SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    store = zarr.DirectoryStore(filepath)
    root = zarr.open_group(store, mode='r+')
    root['data'][start:end] = arr[start:end]

    shm.close()


def _read_chunk_worker(args):
    filepath, shm_name, shape, dtype, chunk_rows, chunk_idx = args
    start = chunk_idx * chunk_rows
    end = min(start + chunk_rows, shape[0])

    store = zarr.DirectoryStore(filepath)
    root = zarr.open_group(store, mode='r')
    chunk_data = root['data'][start:end]  # décompression LZ4 ici (CPU-bound)

    shm = SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr[start:end] = chunk_data  # write into shared mem
    shm.close()



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

    def drop(self, cols):
        """
        Remove columns whose names are in `cols`.
        Keeps original column order.
        """
        if isinstance(cols, str):
            cols = [cols]

        # vérification
        missing = [c for c in cols if c not in self.colName]
        if missing:
            raise KeyError(f"Columns not found: {missing}")

        # colonnes conservées dans l'ordre original
        keep_idx = [i for i, c in enumerate(self.colName) if c not in cols]
        keep_cols = [self.colName[i] for i in keep_idx]

        # CAS 1 : structure classique (N, F, ...)
        if self.data.dim() >= 2:
            new_data = self.data[:, keep_idx]

        # CAS 2 : série avec une seule colonne ND
        elif self.ncols == 1:
            if len(keep_idx) == 0:
                raise ValueError("Cannot drop the only column")
            new_data = self.data

        else:
            raise ValueError("Unsupported tensor shape")

        return serie._from_raw(
            new_data,
            keep_cols,
            self.storingDevice,
            self.storingDtype
        )

    def save(self, filepath, n_threads=None):
        arr = self.data.detach().cpu().numpy()
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)

        bytes_per_row = arr[0].nbytes
        target_chunk_bytes = 4 * 1024 * 1024
        chunk_rows = max(1, min(arr.shape[0], target_chunk_bytes // max(1, bytes_per_row)))
        chunk_shape = (chunk_rows,) + arr.shape[1:]
        n_chunks = (arr.shape[0] + chunk_rows - 1) // chunk_rows

        store = zarr.DirectoryStore(filepath)
        root = zarr.open_group(store, mode='w')
        root.create_dataset(
            'data', shape=arr.shape, chunks=chunk_shape,
            dtype=arr.dtype, compressor=_COMPRESSOR, overwrite=True
        )
        root.attrs['colName'] = json.dumps(self.colName)
        root.attrs['dtype'] = str(self.data.dtype)
        root.attrs['device'] = str(self.data.device)

        if n_threads is None:
            # ── séquentiel : zéro overhead ──────────────────────────────────────
            for i in range(n_chunks):
                start = i * chunk_rows
                end = min(start + chunk_rows, arr.shape[0])
                root['data'][start:end] = arr[start:end]
        else:
            # ── parallèle via shared memory ─────────────────────────────────────
            shm = SharedMemory(create=True, size=arr.nbytes)
            shared = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            np.copyto(shared, arr)

            args = [
                (filepath, shm.name, arr.shape, arr.dtype, chunk_rows, i)
                for i in range(n_chunks)
            ]
            try:
                with ProcessPoolExecutor(max_workers=n_threads) as pool:
                    list(pool.map(_write_chunk_worker, args))
            finally:
                shm.close()
                shm.unlink()

def load(filepath, device=None, dtype=None, n_threads=None):
    # reading small file (1Go in parallel is extremelly slow, but for large files (30Go, it is way faster)
    store = zarr.DirectoryStore(filepath)
    root  = zarr.open_group(store, mode='r')
    z     = root['data']

    shape      = z.shape
    np_dtype   = z.dtype
    chunk_rows = z.chunks[0]
    n_chunks   = (shape[0] + chunk_rows - 1) // chunk_rows

    if n_threads is None:
        # ── séquentiel : lecture directe depuis zarr ─────────────────────────
        arr = z[:]  # zarr gère déjà la décompression chunk par chunk en interne
    else:
        # ── parallèle via shared memory ─────────────────────────────────────
        nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
        shm = SharedMemory(create=True, size=nbytes)

        args = [
            (filepath, shm.name, shape, np_dtype, chunk_rows, i)
            for i in range(n_chunks)
        ]
        try:
            with ProcessPoolExecutor(max_workers=n_threads) as pool:
                list(pool.map(_read_chunk_worker, args))
            arr = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)
            arr = np.array(arr)
        finally:
            shm.close()
            shm.unlink()

    colName = json.loads(root.attrs['colName'])
    saved_dtype  = root.attrs.get('dtype', None)
    saved_device = root.attrs.get('device', None)

    final_dtype  = dtype if dtype is not None else (
        getattr(torch, saved_dtype.split(".")[-1]) if saved_dtype else None
    )
    final_device = device if device is not None else saved_device

    return serie(arr, colName, storingDevice=final_device, storingDtype=final_dtype)


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
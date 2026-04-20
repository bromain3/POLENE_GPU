from POLENE_GPU.dataHandler.serie import serie

import h5py
import torch
import numpy as np
import cantera as ct

def readH5Database(caseName, kinetics, storingDtype=None, storingDevice=None):
    gas = ct.Solution(kinetics)
    spe = gas.species_names

    with h5py.File(caseName, 'r') as f:
        T_X = f['X/T'][:]    # (n_steps, n_cells)
        P_X = f['X/P'][:]
        Y_X = np.stack([f[f'X/Y_{s}'][:] for s in spe], axis=2)   # (n_steps, n_cells, n_spe)

        T_Y = f['Y/T'][:]
        P_Y = f['Y/P'][:]
        Y_Y = np.stack([f[f'Y/Y_{s}'][:] for s in spe], axis=2)

    # Aplatir en (n_steps * n_cells,) — un seul grand batch
    T_X_flat = torch.tensor(T_X.reshape(-1), dtype=storingDtype, device=storingDevice)
    P_X_flat = torch.tensor(P_X.reshape(-1), dtype=storingDtype, device=storingDevice)
    Y_X_flat = torch.tensor(Y_X.reshape(-1, len(spe)), dtype=storingDtype, device=storingDevice)

    T_Y_flat = torch.tensor(T_Y.reshape(-1), dtype=storingDtype, device=storingDevice)
    P_Y_flat = torch.tensor(P_Y.reshape(-1), dtype=storingDtype, device=storingDevice)
    Y_Y_flat = torch.tensor(Y_Y.reshape(-1, len(spe)), dtype=storingDtype, device=storingDevice)


    X_TPY = torch.cat([T_X_flat.unsqueeze(1), P_X_flat.unsqueeze(1), Y_X_flat], dim=1)
    Y_TPY = torch.cat([T_Y_flat.unsqueeze(1), P_Y_flat.unsqueeze(1), Y_Y_flat], dim=1)

    X_TPY = serie(X_TPY, ['T', 'P'] + list(spe), storingDevice=storingDevice, storingDtype=storingDtype)
    Y_TPY = serie(Y_TPY, ['T', 'P'] + list(spe), storingDevice=storingDevice, storingDtype=storingDtype)


    return X_TPY, Y_TPY
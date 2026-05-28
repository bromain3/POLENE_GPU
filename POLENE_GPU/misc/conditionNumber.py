import cantera as ct
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial

# --- État global par worker ---
_worker_gas = None


def _init_condition_worker(kinetics):
    global _worker_gas
    _worker_gas = ct.Solution(kinetics)


def _compute_kappa_chunk(states_batch, reactor_type='cstV'):
    """
    Calcule le condition number pour un batch.
    reactor_type : 'cstV' ou 'cstP'
    """
    global _worker_gas
    N_points = states_batch.shape[0]
    sa = ct.SolutionArray(_worker_gas, shape=(N_points,))

    # 1. État
    sa.TPY = states_batch[:, 0], states_batch[:, 1], states_batch[:, 2:]

    omega = sa.net_production_rates  # kmol/m3/s
    rho = sa.density
    T = sa.T

    # 2. Calcul du Tdot et Cdot selon le réacteur
    if reactor_type == 'cstV':
        # --- Volume Constant ---
        u_molar = sa.partial_molar_int_energies  # J/kmol
        cv_mass = sa.cv_mass  # J/kg/K
        Tdot = - np.sum(u_molar * omega, axis=1) / (rho * cv_mass)

        # À volume constant, dC/dt = omega
        Cdot = omega

    elif reactor_type == 'cstP':
        # --- Pression Constante ---
        h_molar = sa.partial_molar_enthalpies  # J/kmol
        cp_mass = sa.cp_mass  # J/kg/K
        Tdot = - np.sum(h_molar * omega, axis=1) / (rho * cp_mass)

        # À pression constante, le volume change : Vdot/V = Tdot/T + n_dot/n
        # dC/dt = omega - C * (Vdot/V)
        C = sa.concentrations
        sum_omega = np.sum(omega, axis=1)
        sum_C = np.sum(C, axis=1)

        v_expansion = (Tdot / T) + (sum_omega / sum_C)
        Cdot = omega - C * v_expansion[:, None]
    else:
        raise ValueError("reactor_type must be 'cstV' or 'cstP'")

    # 3. JVP : d(omega)/dt = (d_omega/dT * Tdot) + (d_omega/dC * Cdot)
    ddT = sa.net_production_rates_ddT  # (N, n_sp)
    ddCi = sa.net_production_rates_ddCi  # (N, n_sp, n_sp)

    term_T = ddT * Tdot[:, None]
    term_C = np.einsum('nij,nj->ni', ddCi, Cdot)

    jvp = term_T + term_C

    # 4. Kappa
    norm_jvp = np.linalg.norm(jvp, axis=1)
    norm_omega = np.linalg.norm(omega, axis=1)
    kappa = norm_jvp / np.maximum(norm_omega, 1e-12)

    return kappa


def computeConditionNumberParallel(TPY, kinetics, reactor_type='cstV', n_workers=None, chunksize=1500):
    """
    X : TPY (N, 2 + n_sp)
    reactor_type : 'cstV' ou 'cstP'
    """
    if isinstance(TPY, torch.Tensor):
        states = TPY.cpu().detach().numpy()
    elif isinstance(TPY, np.ndarray):
        states = TPY
    else:
        states = TPY.data.cpu().detach().numpy()

    N = states.shape[0]
    if n_workers is None:
        n_workers = mp.cpu_count()

    chunks = [states[i:i + chunksize] for i in range(0, N, chunksize)]

    # Utilisation de partial pour passer le reactor_type au worker
    worker_func = partial(_compute_kappa_chunk, reactor_type=reactor_type)

    print(f"Calcul Kappa ({reactor_type}) sur {N} points...")

    with Pool(
            processes=n_workers,
            initializer=_init_condition_worker,
            initargs=(kinetics,)
    ) as pool:
        results = list(tqdm(
            pool.imap(worker_func, chunks),
            total=len(chunks),
            desc=f"Condition Number {reactor_type}"
        ))

    return np.concatenate(results)
import cantera as ct
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

from POLENE_GPU.dataHandler.serie import serie

# --- État global par worker (initialisé une seule fois) ---
_worker_gas = None
_worker_reactor_class = None
_worker_dt = None
_worker_seq_len = None


def _worker_init_traj(kinetics, reactorType, dt, seq_len):
    """Appelé une seule fois à la création du worker — parse le YAML ici."""
    global _worker_gas, _worker_reactor_class, _worker_dt, _worker_seq_len

    if reactorType == 'cstV':
        _worker_reactor_class = ct.IdealGasReactor
    elif reactorType == 'cstP':
        _worker_reactor_class = ct.IdealGasConstPressureReactor
    else:
        raise ValueError(f"reactorType must be 'cstV' or 'cstP', got '{reactorType}'")

    _worker_gas = ct.Solution(kinetics)
    _worker_dt = dt
    _worker_seq_len = seq_len


def _worker_task_traj(state):
    """
    Intègre un état initial sur seq_len pas de dt secondes, et retourne
    la liste des états successifs [Y_1, ..., Y_seq_len].

    Paramètres
    ----------
    state : np.ndarray, shape (nCols,) — [T, P, Y_1, ..., Y_K]

    Retour
    ------
    traj : np.ndarray, shape (seq_len, nCols)
    """
    global _worker_gas, _worker_reactor_class, _worker_dt, _worker_seq_len

    dt = _worker_dt
    seq_len = _worker_seq_len

    T_ini = float(state[0])
    P_ini = float(state[1])
    Y_ini = state[2:]

    _worker_gas.TPY = T_ini, P_ini, Y_ini
    reactor = _worker_reactor_class(_worker_gas)
    net = ct.ReactorNet([reactor])

    traj = np.empty((seq_len, len(state)), dtype=np.float64)

    t = 0.0
    for k in range(seq_len):
        t += dt
        net.advance(t)
        traj[k] = [reactor.T, reactor.thermo.P] + list(reactor.thermo.Y)

    return traj


# --------------------------------------------------------------------------- #
#  Interface principale                                                        #
# --------------------------------------------------------------------------- #

def generateTrajectoryTargets(
    X,
    seq_len: int,
    dt: float,
    reactorType: str,
    kinetics: str,
    n_workers: int | None = None,
    chunksize: int = 50,
):
    """
    Pour chaque état de X, génère la trajectoire [Y_1, ..., Y_seq_len] en
    intégrant successivement dt secondes à chaque pas (Y_k = état à t = k*dt).

    Paramètres
    ----------
    X           : serie  — base de données d'entrée, shape (N, nCols)
                           colonnes : [T, P, Y_1, ..., Y_K]
    seq_len     : int    — nombre de pas de temps à générer
    dt          : float  — pas de temps (secondes)
    reactorType : str    — 'cstV' ou 'cstP'
    kinetics    : str    — chemin vers le fichier YAML Cantera
    n_workers   : int    — nombre de processus workers (défaut : cpu_count)
    chunksize   : int    — taille des chunks pour imap

    Retour
    ------
    Ys : liste de seq_len objets `serie`, chacun de shape (N, nCols)
         Ys[k] correspond à l'état à t = (k+1)*dt pour chaque sample de X
    """
    col_names = X.colName
    states = X.data.cpu().detach().numpy()   # (N, nCols)
    N = states.shape[0]

    if n_workers is None:
        n_workers = mp.cpu_count()

    with Pool(
        processes=n_workers,
        initializer=_worker_init_traj,
        initargs=(kinetics, reactorType, dt, seq_len),
    ) as pool:
        results = list(tqdm(
            pool.imap(_worker_task_traj, states, chunksize=chunksize),
            total=N,
            desc=f"Generating {seq_len}-step trajectories",
        ))

    # results : liste de N tableaux (seq_len, nCols) -> on veut seq_len tableaux (N, nCols)
    traj_arr = np.stack(results, axis=0)   # (N, seq_len, nCols)

    Ys = [
        serie(traj_arr[:, k, :], col_names, storingDevice=X.storingDevice, storingDtype=X.storingDtype)
        for k in range(seq_len)
    ]

    return Ys
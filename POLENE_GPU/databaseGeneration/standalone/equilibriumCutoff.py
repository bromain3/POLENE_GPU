import cantera as ct
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

from POLENE_GPU.dataHandler.serie import serie


# --- État global par worker ---
_worker_gas = None
_worker_reactor_type_str = None

def _filter_worker_init(kinetics, reactorType):
    """Chargement du mécanisme une seule fois par worker."""
    global _worker_gas, _worker_reactor_type_str
    _worker_gas = ct.Solution(kinetics)
    _worker_reactor_type_str = reactorType


def _compute_equilibrium_T(state, reactorType):
    """Calcule la température d'équilibre pour un état donné."""
    global _worker_gas

    T = float(state[0])
    P = float(state[1])
    Y = state[2:]

    _worker_gas.TPY = T, P, Y

    if reactorType == 'cstV':
        _worker_gas.equilibrate('UV')
    elif reactorType == 'cstP':
        _worker_gas.equilibrate('HP')
    else:
        raise ValueError(f"reactorType must be 'cstV' or 'cstP', got '{reactorType}'")

    return _worker_gas.T


def _filter_worker_task(args):
    """
    Pour un état donné, retourne True si le point est loin de l'équilibre
    (et doit donc être conservé), False sinon.
    """
    global _worker_reactor_type_str

    state, equilibriumTolerance = args

    T = float(state[0])
    Teq = _compute_equilibrium_T(state, _worker_reactor_type_str)

    is_far_from_equilibrium = abs((Teq - T) / Teq) > equilibriumTolerance
    return is_far_from_equilibrium


def filterEquilibrium(X, Y, reactorType, kinetics,
                      equilibriumTolerance=1e-4,
                      n_workers=None, chunksize=100):
    """
    Filtre les samples de X et Y trop proches de l'équilibre thermodynamique.

    Paramètres
    ----------
    X : serie
        Features (shape N x nCols, colonnes : T, P, Y_k...)
    Y : serie
        Targets associées à X (même shape)
    reactorType : str
        'cstV' (UV) ou 'cstP' (HP)
    kinetics : str
        Chemin vers le fichier de mécanisme (ex: 'gri30.yaml')
    equilibriumTolerance : float
        Seuil relatif sur la température : |Teq - T| / Teq > tol pour conserver
    n_workers : int, optional
        Nombre de workers (défaut : cpu_count)
    chunksize : int
        Taille des chunks envoyés aux workers

    Retours
    -------
    X_filtered, Y_filtered : serie
        Base de données sans les points à l'équilibre
    """
    col_names = X.colName
    states = X.data.cpu().detach().numpy()   # (N, nCols)
    targets = Y.data.cpu().detach().numpy()  # (N, nCols)

    if n_workers is None:
        n_workers = mp.cpu_count()

    args = [(state, equilibriumTolerance) for state in states]

    with Pool(
        processes=n_workers,
        initializer=_filter_worker_init,
        initargs=(kinetics, reactorType)
    ) as pool:
        mask = list(tqdm(
            pool.imap(_filter_worker_task, args, chunksize=chunksize),
            total=len(states),
            desc="Filtrage équilibre"
        ))

    mask = np.array(mask, dtype=bool)

    n_removed = (~mask).sum()
    print(f"Points supprimés : {n_removed}/{len(mask)} "
          f"({100 * n_removed / len(mask):.1f} %)")

    X_filtered = serie(
        states[mask],
        col_names,
        storingDevice=X.storingDevice,
        storingDtype=X.storingDtype
    )
    Y_filtered = serie(
        targets[mask],
        col_names,
        storingDevice=Y.storingDevice,
        storingDtype=Y.storingDtype
    )

    return X_filtered, Y_filtered
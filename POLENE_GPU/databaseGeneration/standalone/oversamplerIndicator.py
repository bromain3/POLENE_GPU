import cantera as ct
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

from POLENE_GPU.dataHandler.serie import serie

# --- État global par worker (initialisé une seule fois) ---
_worker_gas_start = None
_worker_gas_mid = None
_worker_gas_targ = None
_worker_reactor_class = None
_worker_dt = None


def _worker_init(kinetics, reactorType, dt, rtol=1e-9, atol=1e-15):
    """Appelé une seule fois à la création du worker — parse le YAML ici."""
    global _worker_gas_start, _worker_gas_mid, _worker_gas_targ, _worker_reactor_class, _worker_dt, _worker_rtol, _worker_atol

    if reactorType == 'cstV':
        _worker_reactor_class = ct.IdealGasReactor
    elif reactorType == 'cstP':
        _worker_reactor_class = ct.IdealGasConstPressureReactor
    else:
        raise ValueError(f"reactorType must be 'cstV' or 'cstP', got '{reactorType}'")

    _worker_gas_start = ct.Solution(kinetics)
    _worker_gas_mid   = ct.Solution(kinetics)
    _worker_gas_targ  = ct.Solution(kinetics)
    _worker_dt = dt
    _worker_rtol = rtol
    _worker_atol = atol


def _worker_task(args):
    """
    Tâche par tirage.

    Paramètres
    ----------
    args : tuple
        (state, dt_prime)
        - state    : np.ndarray, shape (nCols,) — point de départ X tiré de la base
        - dt_prime : float — instant intermédiaire tiré uniformément dans [0, dt]

    Retour
    ------
    (x_prime, y_prime) : deux np.ndarray de shape (nCols,)
        - x_prime : état du réacteur initialisé en X et intégré jusqu'à dt'
        - y_prime : état du réacteur initialisé en X' et intégré pendant dt secondes
    """
    global _worker_gas_start, _worker_gas_mid, _worker_gas_targ, _worker_reactor_class, _worker_dt, _worker_rtol, _worker_atol

    state, dt_prime = args
    dt = _worker_dt

    T_ini = float(state[0])
    P_ini = float(state[1])
    Y_ini = state[2:]

    # ------------------------------------------------------------------ #
    # Étape 1 : intégrer X → X'  (durée dt')                             #
    # ------------------------------------------------------------------ #
    _worker_gas_mid.TPY = T_ini, P_ini, Y_ini
    reactor_mid = _worker_reactor_class(_worker_gas_mid)
    net_mid = ct.ReactorNet([reactor_mid])
    net_mid.rtol = _worker_rtol
    net_mid.atol = _worker_atol

    if dt_prime > 0.0:
        net_mid.advance(dt_prime)

    x_prime = np.array(
        [reactor_mid.T, reactor_mid.thermo.P] + list(reactor_mid.thermo.Y)
    )

    # ------------------------------------------------------------------ #
    # Étape 2 : intégrer X' → Y'  (durée dt)                             #
    # ------------------------------------------------------------------ #
    _worker_gas_targ.TPY = reactor_mid.T, reactor_mid.thermo.P, reactor_mid.thermo.Y
    reactor_targ = _worker_reactor_class(_worker_gas_targ)
    net_targ = ct.ReactorNet([reactor_targ])
    net_targ.rtol = _worker_rtol
    net_targ.atol = _worker_atol
    net_targ.advance(dt)

    y_prime = np.array(
        [reactor_targ.T, reactor_targ.thermo.P] + list(reactor_targ.thermo.Y)
    )

    return x_prime, y_prime


def _sample_indices(indicator: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Tire n indices selon une densité de probabilité proportionnelle à `indicator`.

    L'indicateur n'est pas normalisé : on le normalise ici avant le tirage.
    Les valeurs négatives ou nulles sont ignorées (poids nul).

    Paramètres
    ----------
    indicator : np.ndarray, shape (N,)
    n         : int — nombre de tirages souhaités
    rng       : générateur numpy

    Retour
    ------
    indices : np.ndarray, shape (n,), dtype int
    """
    weights = np.maximum(indicator, 0.0).astype(np.float64)
    total = weights.sum()
    if total == 0.0:
        raise ValueError(
            "La somme des indicateurs est nulle : impossible de définir une distribution."
        )
    probs = weights / total
    return rng.choice(len(indicator), size=n, replace=True, p=probs)


# --------------------------------------------------------------------------- #
#  Interface principale                                                        #
# --------------------------------------------------------------------------- #

def oversamplingIndicator(
    X,
    indicator: np.ndarray,
    n: int,
    dt: float,
    reactorType: str,
    kinetics: str,
    n_workers: int | None = None,
    chunksize: int = 50,
    seed: int | None = None,
):
    """
    Génère n nouveaux couples (X', Y') par tirage pondéré et intégration décalée.

    Algorithme
    ----------
    1. Tirer n indices dans la base X selon la densité définie par `indicator`.
    2. Pour chaque indice i :
       a. Tirer dt' ~ Uniforme(0, dt).
       b. Intégrer le réacteur initialisé en X[i] jusqu'à t = dt'  → X'_i.
       c. Intégrer le réacteur initialisé en X'_i pendant dt secondes → Y'_i.
    3. Retourner (X_out, Y_out) sous forme de deux objets `serie`.

    Paramètres
    ----------
    X           : serie  — base de données d'entrée, shape (N, nCols)
                           colonnes : [T, P, Y_1, ..., Y_K]
    indicator   : np.ndarray, shape (N,)
                           scalaire par sample, interprété comme densité de probabilité
                           (non normalisé, valeurs ≥ 0)
    n           : int    — nombre de nouveaux samples à générer
    dt          : float  — pas de temps cible (secondes)
    reactorType : str    — 'cstV' ou 'cstP'
    kinetics    : str    — chemin vers le fichier YAML Cantera
    n_workers   : int    — nombre de processus workers (défaut : cpu_count)
    chunksize   : int    — taille des chunks pour imap
    seed        : int    — graine du générateur aléatoire (reproductibilité)

    Retour
    ------
    X_out, Y_out : deux objets `serie` de shape (n, nCols)
    """
    col_names = X.colName
    states = X.data.cpu().detach().numpy()          # (N, nCols)

    N = states.shape[0]
    if len(indicator) != N:
        raise ValueError(
            f"indicator doit avoir la même longueur que X ({N}), "
            f"reçu {len(indicator)}."
        )

    rng = np.random.default_rng(seed)

    # --- Tirage pondéré des indices ---
    chosen_indices = _sample_indices(np.asarray(indicator), n, rng)

    # --- Tirage uniforme de dt' pour chaque sample ---
    dt_primes = rng.uniform(0.0, dt, size=n)

    # --- Construction de la liste de tâches ---
    tasks = [
        (states[idx], dt_primes[k])
        for k, idx in enumerate(chosen_indices)
    ]

    if n_workers is None:
        n_workers = mp.cpu_count()

    X_new_list = []
    Y_new_list = []

    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(kinetics, reactorType, dt),
    ) as pool:
        results = list(tqdm(
            pool.imap(_worker_task, tasks, chunksize=chunksize),
            total=n,
            desc="Weighted resampling",
        ))

    for x_prime, y_prime in results:
        X_new_list.append(x_prime)
        Y_new_list.append(y_prime)

    X_arr = np.array(X_new_list)   # (n, nCols)
    Y_arr = np.array(Y_new_list)   # (n, nCols)

    X_out = serie(X_arr, col_names, storingDevice=X.storingDevice, storingDtype=X.storingDtype)
    Y_out = serie(Y_arr, col_names, storingDevice=X.storingDevice, storingDtype=X.storingDtype)

    return X_out, Y_out

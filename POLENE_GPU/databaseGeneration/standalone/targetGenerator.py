"""
targetGeneration.py

Génère les targets Y à partir d'une base d'états X, par intégration directe
Cantera, selon des paramètres définis par l'utilisateur.

C'est la brique "labels Y" de randomizedDatasetGeneration.py, isolée et
généralisée :
    - reactorType : 'cstP' (IdealGasConstPressureReactor) ou 'cstV' (IdealGasReactor)
    - dt          : un pas de temps (float) OU une liste de temps (Y à plusieurs horizons)
    - rtol / atol : tolérances du solveur Cantera
    - on_failure  : comportement si l'intégration échoue pour un état
    - renormalize_Y : nettoyage optionnel de Y avant intégration
    - n_workers / chunksize : parallélisation

Entrées / sorties (identiques à randomizedDatasetGeneration) :
    X     : `serie`, colonnes [T, P, Y_1, ..., Y_K]
    sortie: (X_out, Y_out), deux `serie` de même format que X
            (si dt est une liste : Y_out est une liste de `serie`, une par dt)

Exemple
-------
    X_out, Y_out = targetGeneration(
        X, kinetics="gri30.yaml", dt=1e-6, reactorType="cstP",
    )

    # Plusieurs horizons de temps (temps absolus depuis l'état initial)
    X_out, Y_list = targetGeneration(
        X, kinetics="gri30.yaml", dt=[1e-7, 1e-6, 1e-5], reactorType="cstV",
        rtol=1e-10, atol=1e-16, on_failure="drop",
    )
"""

import numpy as np
import cantera as ct
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

from POLENE_GPU.dataHandler.serie import serie


# --------------------------------------------------------------------------- #
#  Worker d'intégration                                                       #
# --------------------------------------------------------------------------- #

_gas = None
_reactor_class = None
_times = None
_rtol = None
_atol = None
_perm = None            # indices (dans le mécanisme) des espèces de X, dans l'ordre de X
_renormalize = False


def _worker_init(kinetics, reactorType, times, rtol, atol, perm, renormalize):
    global _gas, _reactor_class, _times, _rtol, _atol, _perm, _renormalize
    if reactorType == 'cstV':
        _reactor_class = ct.IdealGasReactor
    elif reactorType == 'cstP':
        _reactor_class = ct.IdealGasConstPressureReactor
    else:
        raise ValueError(f"reactorType doit être 'cstV' ou 'cstP', reçu '{reactorType}'")
    _gas = ct.Solution(kinetics)
    _times = times
    _rtol = rtol
    _atol = atol
    _perm = perm
    _renormalize = renormalize


def _worker_task(state):
    """
    Intègre UN état [T, P, Y_1..Y_K] et retourne un tableau (n_dt, 2+K)
    contenant [T, P, Y_1..Y_K] à chaque temps demandé, ou None en cas d'échec.
    """
    T0, P0 = float(state[0]), float(state[1])
    Y0 = np.asarray(state[2:], dtype=np.float64)

    if _renormalize:
        Y0 = np.clip(Y0, 0.0, None)
        s = Y0.sum()
        if s <= 0:
            return None
        Y0 = Y0 / s

    # Y dans l'ordre du mécanisme (au cas où l'ordre des colonnes de X diffère)
    Y_gas = np.zeros(_gas.n_species)
    Y_gas[_perm] = Y0

    try:
        _gas.TPY = T0, P0, Y_gas
        reactor = _reactor_class(_gas)
        net = ct.ReactorNet([reactor])
        net.rtol = _rtol
        net.atol = _atol

        out = np.empty((len(_times), 2 + len(_perm)))
        for k, t in enumerate(_times):
            net.advance(t)
            out[k, 0] = reactor.T
            out[k, 1] = reactor.thermo.P
            out[k, 2:] = reactor.thermo.Y[_perm]
    except ct.CanteraError:
        return None

    if not np.all(np.isfinite(out)):
        return None
    return out


# --------------------------------------------------------------------------- #
#  Fonction principale                                                        #
# --------------------------------------------------------------------------- #

def targetGeneration(
    X,
    kinetics: str,
    dt,
    reactorType: str,
    rtol: float = 1e-9,            # défaut Cantera
    atol: float = 1e-15,           # défaut Cantera
    renormalize_Y: bool = False,
    on_failure: str = 'nan',       # 'nan' | 'drop' | 'raise'
    n_workers: int | None = None,
    chunksize: int = 50,
    verbose: bool = True,
):
    """
    Calcule les targets Y associées à la base X par intégration Cantera.

    Paramètres
    ----------
    X : serie
        États d'entrée, colonnes [T, P, Y_1, ..., Y_K]. Les noms de colonnes
        des espèces (X.colName[2:]) doivent correspondre à ceux du mécanisme
        (l'ordre peut différer, il est géré automatiquement).
    kinetics : str
        Chemin vers le mécanisme Cantera (YAML).
    dt : float ou liste de float
        Temps d'intégration depuis l'état initial (s). Une liste doit être
        strictement croissante ; on obtient alors un Y par élément.
    reactorType : str
        'cstP' (pression constante) ou 'cstV' (volume constant).
    rtol, atol : float
        Tolérances relative / absolue du solveur (ReactorNet).
    renormalize_Y : bool
        Si True, Y est mis à zéro là où il est négatif puis renormalisé
        (somme = 1) avant l'intégration. Par défaut False : Cantera se
        contente de normaliser la somme.
    on_failure : str
        Que faire si l'intégration d'un état échoue :
          - 'nan'   : la ligne est conservée, Y = NaN ;
          - 'drop'  : la ligne est retirée de X_out ET de Y_out ;
          - 'raise' : lève une RuntimeError.
    n_workers, chunksize :
        Parallélisation (multiprocessing.Pool). n_workers=None -> tous les CPU.
    verbose : bool
        Barre de progression et résumé des échecs.

    Retour
    ------
    X_out : serie
        États d'entrée (filtrés si on_failure='drop'), non modifiés par
        renormalize_Y.
    Y_out : serie ou list[serie]
        Targets. Une `serie` si dt est un scalaire, une liste de `serie`
        (une par dt) sinon.
    """
    # --- validation des paramètres ---
    if reactorType not in ('cstV', 'cstP'):
        raise ValueError(f"reactorType doit être 'cstV' ou 'cstP', reçu '{reactorType}'")
    if on_failure not in ('nan', 'drop', 'raise'):
        raise ValueError(f"on_failure doit être 'nan', 'drop' ou 'raise', reçu '{on_failure}'")

    scalar_dt = np.isscalar(dt)
    times = [float(dt)] if scalar_dt else [float(t) for t in dt]
    if len(times) == 0:
        raise ValueError("dt ne peut pas être une liste vide")
    if times[0] <= 0 or any(b <= a for a, b in zip(times[:-1], times[1:])):
        raise ValueError("dt doit être strictement positif (et strictement croissant si liste)")

    # --- lecture de X et correspondance espèces X <-> mécanisme ---
    col_names = X.colName
    states = X.data.cpu().detach().numpy()          # (N, 2+K)
    N = states.shape[0]
    species_names = list(col_names[2:])

    gas_ref = ct.Solution(kinetics)
    if set(species_names) != set(gas_ref.species_names):
        missing = set(gas_ref.species_names) - set(species_names)
        unknown = set(species_names) - set(gas_ref.species_names)
        raise ValueError(
            "Les espèces de X ne correspondent pas au mécanisme. "
            f"Absentes de X : {sorted(missing)} ; inconnues du mécanisme : {sorted(unknown)}"
        )
    perm = np.array([gas_ref.species_index(n) for n in species_names])

    if n_workers is None:
        n_workers = mp.cpu_count()

    # --- intégration parallèle ---
    with Pool(processes=n_workers, initializer=_worker_init,
              initargs=(kinetics, reactorType, times, rtol, atol, perm, renormalize_Y)) as pool:
        results = list(tqdm(
            pool.imap(_worker_task, list(states), chunksize=chunksize),
            total=N,
            desc="Intégration Cantera (labels Y)",
            disable=not verbose,
        ))

    # --- assemblage : (N, n_dt, 2+K), NaN pour les échecs ---
    n_dt = len(times)
    Y_arr = np.full((N, n_dt, states.shape[1]), np.nan)
    failed = np.zeros(N, dtype=bool)
    for i, res in enumerate(results):
        if res is None:
            failed[i] = True
        else:
            Y_arr[i] = res

    n_failed = int(failed.sum())
    if verbose and n_failed > 0:
        print(f"[targetGeneration] {n_failed}/{N} intégrations échouées "
              f"(on_failure='{on_failure}')")

    if n_failed > 0 and on_failure == 'raise':
        raise RuntimeError(
            f"{n_failed} intégrations ont échoué (indices : {np.where(failed)[0][:20].tolist()}...)"
        )

    keep = ~failed if on_failure == 'drop' else np.ones(N, dtype=bool)

    # --- construction des `serie` de sortie ---
    X_out = serie(states[keep], col_names,
                  storingDevice=X.storingDevice, storingDtype=X.storingDtype)
    Y_list = [
        serie(Y_arr[keep, k, :], col_names,
              storingDevice=X.storingDevice, storingDtype=X.storingDtype)
        for k in range(n_dt)
    ]

    return X_out, (Y_list[0] if scalar_dt else Y_list)
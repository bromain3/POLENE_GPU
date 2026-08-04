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


def _worker_init(kinetics, reactorType, dt):
    """Appelé une seule fois à la création du worker — parse le YAML ici."""
    global _worker_gas, _worker_reactor_class, _worker_dt

    if reactorType == 'cstV':
        _worker_reactor_class = ct.IdealGasReactor
    elif reactorType == 'cstP':
        _worker_reactor_class = ct.IdealGasConstPressureReactor
    else:
        raise ValueError(f"reactorType must be 'cstV' or 'cstP', got '{reactorType}'")

    _worker_gas = ct.Solution(kinetics)
    _worker_dt = dt


def _worker_task(state):
    """
    Intègre un état X pendant dt secondes et retourne Y.

    Paramètres
    ----------
    state : np.ndarray, shape (nCols,) — [T, P, Y_1, ..., Y_K]

    Retour
    ------
    y : np.ndarray, shape (nCols,)
    """
    global _worker_gas, _worker_reactor_class, _worker_dt

    T_ini = float(state[0])
    P_ini = float(state[1])
    Y_ini = state[2:]

    _worker_gas.TPY = T_ini, P_ini, Y_ini
    reactor = _worker_reactor_class(_worker_gas)
    net = ct.ReactorNet([reactor])
    net.advance(_worker_dt)

    y = np.array([reactor.T, reactor.thermo.P] + list(reactor.thermo.Y))
    return y


# --------------------------------------------------------------------------- #
#  Tirage dans la box (log-uniforme + contrainte simplex sur les espèces)      #
# --------------------------------------------------------------------------- #

def _compute_bounds(states: np.ndarray, margin_percent: float):
    """
    Calcule les bornes [lo, hi] par colonne, étendues de margin_percent,
    et clippées à 0 si négatives.
    """
    lo_data = states.min(axis=0)
    hi_data = states.max(axis=0)
    span = hi_data - lo_data
    # marge relative au span (si span nul, marge relative à la valeur elle-même)
    margin_abs = np.where(span > 0, span, np.abs(lo_data)) * (margin_percent / 100.0)

    lo = lo_data - margin_abs
    hi = hi_data + margin_abs

    lo = np.maximum(lo, 0.0)
    hi = np.maximum(hi, 0.0)
    # sécurité : si la marge a fait passer hi sous lo (span nul + lo=0), on force hi >= lo
    hi = np.maximum(hi, lo)

    return lo, hi


def _sample_batch(
    batch_size: int,
    lo: np.ndarray,
    hi: np.ndarray,
    log_mask: np.ndarray,
    log_floor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Tire un batch (batch_size, nCols) de valeurs indépendantes par colonne,
    en respectant le mode (log-uniforme ou uniforme) de chaque colonne.
    Ne gère PAS la contrainte simplex — c'est fait à l'étage au-dessus.
    """
    ncols = lo.shape[0]
    out = np.empty((batch_size, ncols), dtype=np.float64)

    for j in range(ncols):
        lo_j, hi_j = lo[j], hi[j]

        if hi_j <= log_floor:
            # box dégénérée (colonne quasi toujours nulle) : on force à 0
            out[:, j] = 0.0
            continue

        if log_mask[j]:
            lo_eff = max(lo_j, log_floor)
            log_lo, log_hi = np.log(lo_eff), np.log(hi_j)
            out[:, j] = np.exp(rng.uniform(log_lo, log_hi, size=batch_size))
        else:
            out[:, j] = rng.uniform(lo_j, hi_j, size=batch_size)

    return out


def _resample_species_with_rejection(
    candidates: np.ndarray,
    species_slice: slice,
    lo: np.ndarray,
    hi: np.ndarray,
    log_mask: np.ndarray,
    log_floor: float,
    rng: np.random.Generator,
    max_attempts: int,
):
    """
    Renormalise les espèces pour sum=1, et ré-échantillonne (par rejet)
    les lignes dont la renormalisation sort de la box.

    Retour
    ------
    candidates : np.ndarray — modifié en place, colonnes espèces renormalisées
    n_fallback : int — nombre de lignes ayant atteint max_attempts (clip + renorm forcé)
    """
    n = candidates.shape[0]
    species_lo = lo[species_slice]
    species_hi = hi[species_slice]

    def renorm(mat):
        s = mat.sum(axis=1, keepdims=True)
        s = np.where(s <= 0, 1.0, s)  # évite division par zéro (cas dégénéré)
        return mat / s

    candidates[:, species_slice] = renorm(candidates[:, species_slice])

    def in_box(mat):
        return np.all((mat >= species_lo) & (mat <= species_hi), axis=1)

    pending = ~in_box(candidates[:, species_slice])
    n_fallback = 0
    attempt = 0

    while np.any(pending) and attempt < max_attempts:
        n_pending = int(pending.sum())
        new_species = _sample_batch(
            n_pending, lo[species_slice], hi[species_slice],
            log_mask[species_slice], log_floor, rng,
        )
        new_species = renorm(new_species)
        candidates[pending, species_slice] = new_species
        pending = ~in_box(candidates[:, species_slice])
        attempt += 1

    if np.any(pending):
        n_fallback = int(pending.sum())
        clipped = np.clip(candidates[pending][:, species_slice], species_lo, species_hi)
        candidates[pending, species_slice] = renorm(clipped)

    return candidates, n_fallback


def sampleBoxAugmentation(
    X,
    margin_percent: float,
    n: int,
    dt: float,
    reactorType: str,
    kinetics: str,
    excluded_columns: list[str] | None = None,
    log_floor: float = 1e-30,
    max_attempts: int = 20,
    n_workers: int | None = None,
    chunksize: int = 50,
    seed: int | None = None,
):
    """
    Génère n nouveaux couples (X, Y) par tirage aléatoire dans une box
    étendue autour de la base de données, puis intégration Cantera.

    Algorithme
    ----------
    1. Calcul des bornes [lo, hi] par colonne = [min, max] de la base
       étendues de margin_percent, clippées à 0 si négatives.
    2. Tirage indépendant par colonne :
       - colonnes dans `excluded_columns` (T, P par défaut) : uniforme.
       - autres colonnes (espèces par défaut) : log-uniforme, avec un
         plancher `log_floor` pour gérer les bornes inférieures nulles.
    3. Renormalisation des colonnes espèces pour que leur somme vaille 1.
    4. Rejet vectorisé : les lignes dont la renormalisation sort de la box
       espèces sont ré-échantillonnées (par batch), jusqu'à convergence ou
       `max_attempts` ; au-delà, fallback = clip + renormalisation forcée.
    5. Intégration Cantera de chaque X généré pendant dt secondes → Y.

    Paramètres
    ----------
    X                : serie — base de données de référence, shape (N, nCols)
                       colonnes : [T, P, Y_1, ..., Y_K], espèces sommant à 1
    margin_percent   : float — marge relative appliquée aux bornes (en %)
    n                : int   — nombre de nouveaux samples à générer
    dt               : float — durée d'intégration (secondes)
    reactorType      : str   — 'cstV' ou 'cstP'
    kinetics         : str   — chemin vers le fichier YAML Cantera
    excluded_columns : list[str] | None — colonnes tirées en uniforme plutôt
                       qu'en log-uniforme. Défaut : les deux premières colonnes
                       (T, P).
    log_floor        : float — plancher utilisé en espace log pour les bornes
                       inférieures nulles (typiquement une fraction massique
                       négligeable, ex: 1e-30)
    max_attempts     : int   — nombre max de tours de rejet avant fallback
    n_workers        : int   — nombre de processus workers (défaut : cpu_count)
    chunksize        : int   — taille des chunks pour imap
    seed             : int   — graine du générateur aléatoire (reproductibilité)

    Retour
    ------
    X_out, Y_out : deux objets `serie` de shape (n, nCols)
    n_fallback   : int — nombre de samples ayant nécessité un fallback
                   (clip + renorm forcé) plutôt qu'un rejet propre — un
                   nombre élevé signale une marge trop serrée vis-à-vis
                   du nombre d'espèces.
    """
    col_names = X.colName
    states = X.data.cpu().detach().numpy()  # (N, nCols)
    ncols = states.shape[1]

    if excluded_columns is None:
        excluded_columns = list(col_names[:2])  # T, P par défaut

    excluded_idx = {col_names.index(c) for c in excluded_columns}
    log_mask = np.array([i not in excluded_idx for i in range(ncols)])

    lo, hi = _compute_bounds(states, margin_percent)

    rng = np.random.default_rng(seed)

    # --- tirage initial de tout le batch (T, P, espèces indépendamment) ---
    X_new = _sample_batch(n, lo, hi, log_mask, log_floor, rng)

    # --- renormalisation + rejet vectorisé sur les colonnes espèces ---
    species_slice = slice(2, ncols)
    X_new, n_fallback = _resample_species_with_rejection(
        X_new, species_slice, lo, hi, log_mask, log_floor, rng, max_attempts,
    )

    # --- calcul de Y via Cantera ---
    if n_workers is None:
        n_workers = mp.cpu_count()

    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(kinetics, reactorType, dt),
    ) as pool:
        Y_list = list(tqdm(
            pool.imap(_worker_task, list(X_new), chunksize=chunksize),
            total=n,
            desc="Box sampling + integration",
        ))

    Y_arr = np.array(Y_list)  # (n, nCols)

    X_out = serie(X_new, col_names, storingDevice=X.storingDevice, storingDtype=X.storingDtype)
    Y_out = serie(Y_arr, col_names, storingDevice=X.storingDevice, storingDtype=X.storingDtype)

    if n_fallback > 0:
        print(f"[sampleBoxAugmentation] Attention : {n_fallback}/{n} samples "
              f"ont nécessité un fallback (clip + renorm forcé).")

    return X_out, Y_out, n_fallback
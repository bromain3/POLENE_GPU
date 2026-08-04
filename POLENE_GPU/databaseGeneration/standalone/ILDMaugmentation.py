import cantera as ct
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

from POLENE_GPU.dataHandler.serie import serie

# --------------------------------------------------------------------------- #
#  Workers Cantera pour l'intégration exacte (identiques à sampleBoxAugmentation) #
# --------------------------------------------------------------------------- #

_worker_gas = None
_worker_reactor_class = None
_worker_dt = None


def _worker_init(kinetics, reactorType, dt):
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
    """state : np.ndarray, shape (nCols,) — [T, P, Y_1, ..., Y_K]"""
    global _worker_gas, _worker_reactor_class, _worker_dt

    T_ini = float(state[0])
    P_ini = float(state[1])
    Y_ini = state[2:]

    _worker_gas.TPY = T_ini, P_ini, Y_ini
    reactor = _worker_reactor_class(_worker_gas)
    net = ct.ReactorNet([reactor])
    net.advance(_worker_dt)

    return np.array([reactor.T, reactor.thermo.P] + list(reactor.thermo.Y))


# --------------------------------------------------------------------------- #
#  Décomposition rapide / lente de la Jacobienne                              #
# --------------------------------------------------------------------------- #

def _real_eigenbasis(J: np.ndarray):
    """
    Décompose J (réelle, shape (d, d)) en valeurs propres et une base REELLE
    de vecteurs (utile car les vecteurs propres associés à des valeurs
    propres complexes conjuguées ne sont pas réels individuellement, mais
    span un sous-espace réel de dimension 2).

    Retour
    ------
    lam_real : np.ndarray, shape (d,) — partie réelle des valeurs propres,
               répétée de façon cohérente avec les colonnes de `basis`
    basis    : np.ndarray, shape (d, d) — colonnes = vecteurs réels formant
               une base du même espace que les vecteurs propres (normalisés)
    """
    eigvals, eigvecs = np.linalg.eig(J)
    d = J.shape[0]
    basis = np.zeros((d, d), dtype=np.float64)
    lam_real = np.zeros(d, dtype=np.float64)

    j = 0
    used = np.zeros(d, dtype=bool)
    for i in range(d):
        if used[i]:
            continue
        lam = eigvals[i]
        v = eigvecs[:, i]
        if np.abs(lam.imag) < 1e-10:
            # valeur propre réelle
            vr = v.real
            norm = np.linalg.norm(vr)
            basis[:, j] = vr / norm if norm > 0 else vr
            lam_real[j] = lam.real
            used[i] = True
            j += 1
        else:
            # cherche le conjugué correspondant
            conj_idx = None
            for k in range(i + 1, d):
                if not used[k] and np.isclose(eigvals[k], np.conj(lam), atol=1e-8):
                    conj_idx = k
                    break
            used[i] = True
            if conj_idx is not None:
                used[conj_idx] = True
            # base réelle du plan invariant : Re(v), Im(v)
            vr, vi = v.real, v.imag
            for vec in (vr, vi):
                if j >= d:
                    break
                norm = np.linalg.norm(vec)
                basis[:, j] = vec / norm if norm > 0 else vec
                lam_real[j] = lam.real  # même partie réelle (paire conjuguée)
                j += 1

    return lam_real[:j], basis[:, :j]


def _split_fast_slow(lam_real: np.ndarray, n_slow: int | None, gap_ratio: float):
    """
    Détermine quels modes sont "rapides" (à perturber) vs "lents" (à préserver,
    tangents à la trajectoire chimique).

    - Si n_slow est fourni : on protège explicitement les n_slow modes de plus
      petit |Re(lambda)| (les plus lents), tous les autres sont "rapides".
    - Sinon : détection automatique du plus grand saut (en log) dans les
      |Re(lambda)| triés ; tout ce qui est sous le saut est "lent".
      gap_ratio est le facteur minimal exigé pour valider un saut ; si aucun
      saut ne dépasse ce facteur, tous les modes sont considérés rapides
      (comportement conservateur : mieux vaut sous-protéger que perturber
      dans le vide si le mécanisme n'a pas de séparation d'échelle nette).

    Retour : mask booléen (True = mode rapide, à perturber)
    """
    d = lam_real.shape[0]
    mag = np.abs(lam_real)
    order = np.argsort(mag)  # croissant

    if n_slow is not None:
        mask = np.ones(d, dtype=bool)
        mask[order[:n_slow]] = False
        return mask

    mag_sorted = mag[order]
    eps = 1e-30
    log_mag = np.log(np.maximum(mag_sorted, eps))
    diffs = np.diff(log_mag)
    if len(diffs) == 0:
        return np.ones(d, dtype=bool)

    k = np.argmax(diffs)  # plus grand saut
    if np.exp(diffs[k]) < gap_ratio:
        return np.ones(d, dtype=bool)  # pas de séparation nette -> tout rapide

    mask = np.ones(d, dtype=bool)
    mask[order[: k + 1]] = False  # tout ce qui est avant/au saut = lent
    return mask


# --------------------------------------------------------------------------- #
#  Génération d'un point perturbé physiquement admissible                     #
# --------------------------------------------------------------------------- #

def _sample_one_perturbed_state(
    y_star: np.ndarray,
    J: np.ndarray,
    state_idx: np.ndarray,
    species_pos_in_state: np.ndarray,
    sigma0: float,
    dt: float,
    n_slow: int | None,
    gap_ratio: float,
    max_backtrack: int,
    backtrack_factor: float,
    min_fraction: float,
    rng: np.random.Generator,
    convergence_tol: float = 1e-10,
):
    """
    Construit un état perturbé physiquement admissible à partir de y_star.

    y_star               : état de référence complet, shape (nCols,) = [T,P,Y...]
    J                    : Jacobienne locale, shape (d, d), d = len(state_idx)
    state_idx            : indices (dans y_star) des variables couvertes par J
                           (typiquement tout sauf P)
    species_pos_in_state : indices (dans le sous-état de taille d, PAS dans
                           y_star) qui correspondent aux fractions massiques
    sigma0               : amplitude globale de la perturbation
    dt                   : pas de temps caractéristique (pour pondérer les
                           modes selon |Re(lambda)|*dt)
    n_slow, gap_ratio    : cf. _split_fast_slow
    max_backtrack        : nombre max de réductions d'amplitude si la
                           perturbation viole la physique
    backtrack_factor     : facteur de réduction à chaque tentative (< 1)
    min_fraction         : tolérance négative acceptée avant clip forcé
                           (bruit numérique), typiquement 0.0 ou -1e-12

    Retour
    ------
    y_new     : état perturbé complet, shape (nCols,)
    fallback  : bool — True si le backtracking a échoué et qu'un clip+renorm
                forcé a été nécessaire (signale une amplitude sigma0 trop
                grande par rapport à la distance à la frontière physique)
    """
    lam_real, basis = _real_eigenbasis(J)
    fast_mask = _split_fast_slow(lam_real, n_slow, gap_ratio)

    # écarts-types par mode : modes très raides (grand |Re(lambda)|*dt)
    # -> amplitude réduite (physiquement peu de marge d'écart) ;
    # modes proches de la frontière rapide/lent -> amplitude proche de sigma0.
    sigma_i = np.where(
        fast_mask,
        sigma0 / (1.0 + np.abs(lam_real) * dt),
        0.0,  # modes lents non perturbés (préserve la direction tangente)
    )

    y_sub_star = y_star[state_idx].copy()

    # Tirage UNIQUE du coefficient aléatoire c. Le backtracking qui suit doit
    # rétrécir cette MÊME direction de perturbation (delta = basis @ (c*scale)),
    # jamais re-tirer un nouveau c à variance réduite — sinon on ne fait plus
    # du backtracking mais du rejet répété, qui sélectionne artificiellement
    # les tirages où les espèces-traces sont poussées vers le haut (et donc,
    # par conservation de la masse le long des modes rapides, où l'espèce
    # dominante est systématiquement tirée vers le bas pour compenser).
    c = rng.normal(loc=0.0, scale=sigma_i)

    def build_candidate(scale: float):
        delta = basis @ (c * scale)
        return y_sub_star + delta

    candidate = build_candidate(1.0)
    scale = 1.0
    fallback = False

    # Note : pour une espèce à valeur de référence EXACTEMENT nulle et un
    # vecteur propre "rapide" dense (cas fréquent : mécanismes détaillés,
    # espèces traces), le backtracking fait tendre l'excursion vers 0 sans
    # jamais l'atteindre exactement en nombre fini d'itérations. On accepte
    # donc une violation résiduelle sous `convergence_tol` (négligeable
    # numériquement, nettoyée par le clip final ci-dessous) plutôt que
    # d'exiger une non-négativité stricte qui ne serait jamais satisfaite.
    for _ in range(max_backtrack):
        species = candidate[species_pos_in_state]
        worst_violation = min_fraction - species.min() if species.min() < min_fraction else 0.0
        if worst_violation <= convergence_tol:
            break
        scale *= backtrack_factor
        candidate = build_candidate(scale)
    else:
        # le backtracking n'a pas suffi (violation encore significative) :
        # on le signale, mais on nettoie quand même l'état ci-dessous.
        if species.min() < min_fraction - convergence_tol:
            fallback = True

    # T et P doivent rester strictement positifs (sécurité, rarement actif
    # en pratique si sigma0 est correctement calibré)
    for col in (0, 1):  # 0 = T, 1 = P dans y_star (indices absolus)
        col_local = np.where(state_idx == col)[0]
        if col_local.size > 0 and candidate[col_local[0]] <= 0:
            candidate[col_local[0]] = y_sub_star[col_local[0]]
            fallback = True

    # nettoyage final + renormalisation stricte de la simplex des fractions
    species = np.clip(candidate[species_pos_in_state], 0.0, None)
    s = species.sum()
    species = species / s if s > 0 else y_sub_star[species_pos_in_state]
    candidate[species_pos_in_state] = species

    y_new = y_star.copy()
    y_new[state_idx] = candidate
    return y_new, fallback


# --------------------------------------------------------------------------- #
#  Fonction principale                                                        #
# --------------------------------------------------------------------------- #

def sampleManifoldAugmentation(
    X,
    J: np.ndarray,
    dt: float,
    reactorType: str,
    kinetics: str,
    sigma0: float = 1e-4,
    n_per_point: int = 1,
    state_idx: list[int] | None = None,
    n_slow: int | None = 0,
    gap_ratio: float = 5.0,
    max_backtrack: int = 10,
    backtrack_factor: float = 1.0,
    min_fraction: float = 0.0,
    convergence_tol: float = 1e-10,
    n_workers: int | None = None,
    chunksize: int = 50,
    seed: int | None = None,
):
    """
    Génère des couples (X', Y') par perturbation locale guidée par la
    Jacobienne chimique (modes rapides = normaux à la variété ILDM),
    suivie d'une intégration Cantera exacte.

    Algorithme
    ----------
    1. Pour chaque point de référence y* et sa Jacobienne J associée :
       a. décomposition en valeurs propres réelles / base réelle
          (gère les paires complexes conjuguées) ;
       b. séparation modes rapides (perturbés) / lents (préservés,
          tangents à la trajectoire) ;
       c. tirage gaussien anisotrope le long des modes rapides, amplitude
          pondérée par 1/(1+|Re(lambda)|*dt) ;
       d. correction physique par backtracking (réduction d'amplitude tant
          que des fractions massiques deviennent négatives), puis clip +
          renormalisation stricte de la simplex en dernier recours.
    2. Intégration Cantera exacte de chaque X' sur dt secondes -> Y'
       (aucune approximation linéaire dans la cible).

    Paramètres
    ----------
    X            : serie — base de référence, shape (N, nCols),
                   colonnes [T, P, Y_1, ..., Y_K]
    J            : np.ndarray, shape (N, d, d) — Jacobiennes locales de la
                   cinétique (d(dT/dt, dP/dt, dY/dt)/d(T, P, Y)), cohérentes
                   avec l'état complet [T, P, Y_1, ..., Y_K], une par point
                   de X, dans l'ordre des colonnes désigné par `state_idx`
    dt           : durée d'intégration (secondes), doit correspondre au dt
                   utilisé pour calculer J si vous voulez que la pondération
                   des sigma_i soit cohérente
    reactorType  : 'cstV' ou 'cstP'
    kinetics     : chemin vers le mécanisme YAML Cantera
    sigma0       : amplitude globale de la perturbation (à calibrer sur la
                   dérive one-step réelle de votre réseau)
    n_per_point  : nombre de tirages générés par point de référence
    state_idx    : indices des colonnes de X couvertes par J (dans l'ordre
                   des lignes/colonnes de J). Défaut : toutes les colonnes,
                   i.e. [0, 1, 2, ..., nCols-1] = [T, P, Y_1, ..., Y_K],
                   puisque votre Jacobienne est cohérente avec l'état complet.
                   Si votre J ne couvre en réalité pas P (P traité comme
                   paramètre figé), passez explicitement
                   state_idx=[i for i in range(ncols) if i != 1].
    n_slow       : nombre de modes lents à protéger explicitement (sinon
                   détection automatique du plus grand saut spectral)
    gap_ratio    : facteur minimal du saut spectral pour la détection auto
    max_backtrack, backtrack_factor : cf. _sample_one_perturbed_state
    convergence_tol : tolérance numérique sous laquelle une violation
                   résiduelle de la simplex est considérée comme convergée
                   (évite de compter en fallback des espèces qui tendent
                   vers 0 sans jamais l'atteindre exactement, cf. commentaire
                   dans _sample_one_perturbed_state)
    min_fraction : tolérance négative acceptée avant clip forcé
    n_workers, chunksize, seed : cf. sampleBoxAugmentation

    Retour
    ------
    X_out, Y_out : serie, shape (N * n_per_point, nCols)
    n_fallback   : int — nombre de samples ayant nécessité un clip+renorm
                   forcé (backtracking insuffisant) ; un nombre élevé
                   signale un sigma0 trop grand par rapport à la marge
                   physique disponible localement.
    """
    col_names = X.colName
    states = X.data.cpu().detach().numpy()  # (N, nCols)
    N, ncols = states.shape

    if J.shape[0] != N:
        raise ValueError(f"J doit avoir autant de points que X ({N}), reçu {J.shape[0]}")

    if state_idx is None:
        state_idx = list(range(ncols))  # T, P, et toutes les espèces
    state_idx = np.array(state_idx)

    if J.shape[1] != len(state_idx) or J.shape[2] != len(state_idx):
        raise ValueError(
            f"J doit être de shape (N, {len(state_idx)}, {len(state_idx)}) "
            f"pour correspondre à state_idx, reçu {J.shape[1:]}"
        )

    # positions des espèces DANS le sous-état state_idx (pas dans y_star complet)
    species_cols_full = [i for i in range(ncols) if i >= 2]
    species_pos_in_state = np.array(
        [np.where(state_idx == c)[0][0] for c in species_cols_full if c in state_idx]
    )

    rng = np.random.default_rng(seed)

    X_new = np.empty((N * n_per_point, ncols), dtype=np.float64)
    n_fallback = 0
    row = 0
    for i in tqdm(range(N), desc="Manifold sampling (perturbation)"):
        y_star = states[i]
        J_i = J[i]
        for _ in range(n_per_point):
            y_new, fb = _sample_one_perturbed_state(
                y_star, J_i, state_idx, species_pos_in_state,
                sigma0, dt, n_slow, gap_ratio,
                max_backtrack, backtrack_factor, min_fraction, rng,
                convergence_tol=convergence_tol,
            )
            X_new[row] = y_new
            n_fallback += int(fb)
            row += 1

    # --- intégration Cantera exacte (mêmes workers que sampleBoxAugmentation) ---
    if n_workers is None:
        n_workers = mp.cpu_count()

    n_total = X_new.shape[0]
    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(kinetics, reactorType, dt),
    ) as pool:
        Y_list = list(tqdm(
            pool.imap(_worker_task, list(X_new), chunksize=chunksize),
            total=n_total,
            desc="Manifold sampling + integration",
        ))

    Y_arr = np.array(Y_list)

    X_out = serie(X_new, col_names, storingDevice=X.storingDevice, storingDtype=X.storingDtype)
    Y_out = serie(Y_arr, col_names, storingDevice=X.storingDevice, storingDtype=X.storingDtype)

    if n_fallback > 0:
        print(f"[sampleManifoldAugmentation] Attention : {n_fallback}/{n_total} samples "
              f"ont nécessité un fallback (clip + renorm forcé) -> sigma0 probablement trop grand.")

    return X_out, Y_out, n_fallback
"""
randomizedDatasetGeneration.py

Implémente l'algorithme de randomisation de Liu, Kronenburg & Zirwes (2026),
Section 2.2, en réutilisant la STRUCTURE et les équations de base de Ding,
Readshaw, Rigopoulos & Jones (2021), Combustion and Flame 231, 111493
(Eqs. 5-7), dont Liu et al. explicitent qu'ils s'inspirent directement
("the algorithm developed in this work builds upon the approach by Ding
et al.").

Principe : Ding et al. fournit la STRUCTURE (perturbation d'un potentiel
énergétique + perturbation des concentrations molaires + acceptation/rejet
+ intégration DI), Liu et al. la RAFFINE sur plusieurs points explicites.
Ce script implémente donc la structure de Ding, avec les raffinements de
Liu appliqués partout où celui-ci définit une nouvelle logique ou de
nouvelles valeurs :

RAFFINEMENTS DE LIU ET AL. (2026) PAR RAPPORT A DING ET AL. (2021) :
    1. Sélection préalable de l'état de base (absente chez Ding) : l'état
       de base n'est retenu comme point de départ que s'il respecte déjà
       T in [500, 3000] K et f in [0.00, 0.36] (Section 2.2 de Liu).
    2. Bornes d'acceptation ELARGIES et AVEC BORNE HAUTE sur T (Ding ne
       donne pas de borne haute) : T in [490, 3000] K, f in [0.00, 0.39].
    3. Ratios élémentaires C/H et O/N acceptés avec une TOLERANCE RELATIVE
       de 2% par rapport à l'état de BASE (et non une plage absolue fixe
       comme le Tableau 1 de Ding, qui n'est plus utilisé ici).
    4. La perturbation du potentiel énergétique (Eq. 5 de Ding) porte sur
       l'ENTHALPIE SENSIBLE (et non l'enthalpie totale intégrant
       l'enthalpie de formation) : Liu et al. écrivent explicitement
       "Randomization introduces random variations in SENSIBLE ENTHALPY
       and species mass fractions". L'enthalpie totale est reconstruite
       ensuite comme (enthalpie sensible perturbée) + (enthalpie de
       formation de la NOUVELLE composition), avant de recalculer T.

CE QUI N'EST PAS REDEFINI PAR LIU ET AL. -- ON GARDE DONC LES VALEURS DE
DING ET AL. (2021) PAR DEFAUT :
    - Eq. (5) (transposée à l'enthalpie/énergie SENSIBLE, cf. raffinement 4)
      avec le coefficient a = 8.
    - Eqs. (6)-(7) (perturbation en concentration molaire, exposant
      1 + c/b) avec le coefficient b = 5, strictement identiques à Ding.
    - max_attempts = 10 (nombre de tentatives avant abandon d'un état de
      base).

EXTENSION CONSERVEE PAR RAPPORT A NOS SCRIPTS PRECEDENTS -- cohérence
cstP/cstV :
    - reactorType == 'cstP' : perturbation de l'ENTHALPIE sensible h_sens,
      P fixée à P0, T recalculée via gas.HPY.
    - reactorType == 'cstV' : perturbation de l'ENERGIE INTERNE sensible
      u_sens, volume massique v0 = 1/rho0 fixé, (T, P) recalculées via
      gas.UVY.

EXTENSION AJOUTEE -- choix de la loi de tirage des variables aléatoires
c (Eq. 5) et c_species (Eqs. 6-7) :
    - distribution == 'uniform'  (défaut, identique à Ding et Liu) :
          c ~ U(-1, +1)
    - distribution == 'gaussian' :
          c ~ N(0, 1)
      (écart-type 1, pour rester dans le même ordre de grandeur que la
      loi uniforme d'origine ; ni Ding ni Liu ne spécifient de loi
      gaussienne, c'est une extension).

CE QUI RESTE UNE HYPOTHESE (non précisé ni par Ding ni par Liu) :
    - la température de référence utilisée pour séparer la part
      "sensible" de la part "formation" du potentiel énergétique
      (`T_ref_sensible`, 298.15 K par défaut -- convention standard, mais
      non donnée explicitement dans les deux articles) ;
    - la politique de tirage des états de base dans X (tirage uniforme
      avec remise) ;
    - le calcul des bornes min/max de l'énergie SENSIBLE (Eq. 5) sur
      l'intégralité de X par défaut (`minmax_sample_size` permet de
      sous-échantillonner ce calcul).
"""

import cantera as ct
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

from POLENE_GPU.dataHandler.serie import serie


# --------------------------------------------------------------------------- #
#  Tirage aléatoire : uniforme ou gaussien (extension par rapport à Ding/Liu)#
# --------------------------------------------------------------------------- #

def _draw(rng, distribution, scale, size=None):
    """
    Tire des variables aléatoires c utilisées dans les Eqs. (5)-(7) :
        - distribution == 'uniform'  -> c ~ U(-scale, +scale)  (Ding/Liu, scale=1)
        - distribution == 'gaussian' -> c ~ N(0, scale)        (extension)
    """
    if distribution == 'uniform':
        return rng.uniform(-scale, scale, size=size)
    elif distribution == 'gaussian':
        return rng.normal(0.0, scale, size=size)
    else:
        raise ValueError(
            f"distribution doit être 'uniform' ou 'gaussian', reçu '{distribution}'"
        )

def _perturb_pressure(P, rng, mode, size, distribution, max_resample=100):
    """
    Perturbe un vecteur de pressions P (Pa) pour émuler les variations de
    pression extérieures dues au solveur CFD.
        - mode == 'multiplicative' : P' = P * (1 + size * c)   (size sans unité)
        - mode == 'additive'       : P' = P + size * c         (size en Pa)
    avec c tiré selon `distribution` ('uniform' -> U(-1,1), 'gaussian' -> N(0,1)).
    Les tirages donnant P' <= 0 sont retirés (resampling).
    """
    if mode is None or size == 0.0:
        return P
    if mode not in ('multiplicative', 'additive'):
        raise ValueError(
            f"P_perturbation_mode doit être None, 'multiplicative' ou 'additive', reçu '{mode}'"
        )
    P_new = P.copy()
    todo = np.arange(len(P))
    for _ in range(max_resample):
        if todo.size == 0:
            break
        c = _draw(rng, distribution, 1.0, size=todo.size)
        if mode == 'multiplicative':
            cand = P[todo] * (1.0 + size * c)
        else:
            cand = P[todo] + size * c
        ok = cand > 0.0
        P_new[todo[ok]] = cand[ok]
        todo = todo[~ok]
    if todo.size > 0:
        raise RuntimeError(
            "Impossible d'obtenir des pressions strictement positives : "
            "P_perturbation_size est trop grand."
        )
    return P_new

# --------------------------------------------------------------------------- #
#  Workers pour l'intégration Cantera (labels Y)                             #
# --------------------------------------------------------------------------- #

_int_gas = None
_int_reactor_class = None
_int_dt = None


def _integration_worker_init(kinetics, reactorType, dt):
    global _int_gas, _int_reactor_class, _int_dt
    if reactorType == 'cstV':
        _int_reactor_class = ct.IdealGasReactor
    elif reactorType == 'cstP':
        _int_reactor_class = ct.IdealGasConstPressureReactor
    else:
        raise ValueError(f"reactorType doit être 'cstV' ou 'cstP', reçu '{reactorType}'")
    _int_gas = ct.Solution(kinetics)
    _int_dt = dt


def _integration_worker_task(state):
    """Intègre un état X pendant dt secondes et retourne Y = [T,P,Y_1..Y_K]."""
    global _int_gas, _int_reactor_class, _int_dt
    T_ini, P_ini = float(state[0]), float(state[1])
    Y_ini = state[2:]
    _int_gas.TPY = T_ini, P_ini, Y_ini
    reactor = _int_reactor_class(_int_gas)
    net = ct.ReactorNet([reactor])
    net.advance(_int_dt)
    return np.array([reactor.T, reactor.thermo.P] + list(reactor.thermo.Y))


# --------------------------------------------------------------------------- #
#  Workers pour le calcul des bornes de l'énergie SENSIBLE (Eq. 5 de Ding,   #
#  appliquée à h_sens ou u_sens -- raffinement de Liu)                       #
# --------------------------------------------------------------------------- #

_bounds_gas = None
_bounds_reactorType = None
_bounds_T_ref = None


def _bounds_worker_init(kinetics, reactorType, T_ref):
    global _bounds_gas, _bounds_reactorType, _bounds_T_ref
    _bounds_gas = ct.Solution(kinetics)
    _bounds_reactorType = reactorType
    _bounds_T_ref = T_ref


def _bounds_worker_task(state):
    """Calcule l'énergie SENSIBLE (h_sens en cstP, u_sens en cstV) pour UN état."""
    global _bounds_gas, _bounds_reactorType, _bounds_T_ref
    gas = _bounds_gas
    T0, P0 = float(state[0]), float(state[1])
    Y0 = state[2:]
    gas.TPY = T0, P0, Y0
    e_tot = gas.enthalpy_mass if _bounds_reactorType == 'cstP' else gas.int_energy_mass
    gas.TPY = _bounds_T_ref, P0, Y0
    e_ref = gas.enthalpy_mass if _bounds_reactorType == 'cstP' else gas.int_energy_mass
    return e_tot - e_ref  # énergie sensible


# --------------------------------------------------------------------------- #
#  Workers pour la génération randomisée (structure Ding + raffinements Liu) #
# --------------------------------------------------------------------------- #

_gen_gas = None
_gen_fuel = None
_gen_oxidizer = None
_gen_reactorType = None
_gen_W = None
_gen_sens_min = None
_gen_sens_max = None
_gen_a = None
_gen_b = None
_gen_T_ref = None
_gen_distribution = None


def _generation_worker_init(kinetics, fuel, oxidizer, reactorType, species_names,
                             sens_min, sens_max, a_coef, b_coef, T_ref, distribution):
    global _gen_gas, _gen_fuel, _gen_oxidizer, _gen_reactorType, _gen_W
    global _gen_sens_min, _gen_sens_max, _gen_a, _gen_b, _gen_T_ref, _gen_distribution

    if reactorType not in ('cstV', 'cstP'):
        raise ValueError(f"reactorType doit être 'cstV' ou 'cstP', reçu '{reactorType}'")

    _gen_gas = ct.Solution(kinetics)
    _gen_fuel = fuel
    _gen_oxidizer = oxidizer
    _gen_reactorType = reactorType

    idx_in_gas = [_gen_gas.species_index(name) for name in species_names]
    _gen_W = _gen_gas.molecular_weights[idx_in_gas]

    _gen_sens_min = sens_min
    _gen_sens_max = sens_max
    _gen_a = a_coef
    _gen_b = b_coef
    _gen_T_ref = T_ref
    _gen_distribution = distribution


def _molar_ratio(gas, elem_num, elem_den):
    num = gas.elemental_mole_fraction(elem_num)
    den = gas.elemental_mole_fraction(elem_den)
    return num / den if den > 0 else np.inf


def _generation_worker_task(args):
    """
    Génère UN état randomisé à partir d'un état de base, en suivant la
    structure de Ding et al. (2021, Eqs. 5-7) raffinée par Liu et al.
    (2026, Section 2.2) : sélection préalable, perturbation de l'énergie
    SENSIBLE et des concentrations molaires, acceptation avec tolérance
    relative sur les ratios élémentaires.

    Retourne un np.ndarray [T,P,Y_1..Y_K] si un état a été accepté dans la
    limite de max_attempts, sinon None.
    """
    global _gen_gas, _gen_fuel, _gen_oxidizer, _gen_reactorType, _gen_W
    global _gen_sens_min, _gen_sens_max, _gen_a, _gen_b, _gen_T_ref, _gen_distribution

    (base_state, seed, T_select_range, z_select_range,
     T_accept_range, z_accept_range, elem_ratio_tol, max_attempts) = args

    gas = _gen_gas
    rng = np.random.default_rng(seed)

    T0, P0 = float(base_state[0]), float(base_state[1])
    Y0 = np.clip(np.array(base_state[2:], dtype=np.float64), 0.0, None)

    # --- état de référence : énergie totale/sensible, fraction de       ---
    # --- mélange, ratios élémentaires, volume massique si cstV          ---
    gas.TPY = T0, P0, Y0
    if _gen_reactorType == 'cstP':
        e0_total = gas.enthalpy_mass
        v0 = None
    else:  # 'cstV'
        e0_total = gas.int_energy_mass
        v0 = 1.0 / gas.density

    gas.TPY = _gen_T_ref, P0, Y0
    e0_ref = gas.enthalpy_mass if _gen_reactorType == 'cstP' else gas.int_energy_mass
    e0_sensible = e0_total - e0_ref

    gas.TPY = T0, P0, Y0
    z0 = gas.mixture_fraction(_gen_fuel, _gen_oxidizer, basis='mass')
    CH0 = _molar_ratio(gas, 'C', 'H')
    ON0 = _molar_ratio(gas, 'O', 'N')

    # --- (1) sample selection (Liu) : l'état de base doit déjà respecter ---
    # --- les bornes de sélection                                         ---
    if not (T_select_range[0] <= T0 <= T_select_range[1]):
        return None
    if not (z_select_range[0] <= z0 <= z_select_range[1]):
        return None

    d_sens_range = _gen_sens_max - _gen_sens_min
    y_mole0 = Y0 / _gen_W  # concentrations molaires (kmol/kg), Eq. (6)

    for _ in range(max_attempts):
        # --- Eq. (5) de Ding, appliquée à l'énergie SENSIBLE (Liu) ---
        c_e = float(_draw(rng, _gen_distribution, 1.0))
        e_sens_new = e0_sensible + (c_e / _gen_a) * d_sens_range

        # --- Eqs. (6)-(7) de Ding : perturbation des concentrations      ---
        # --- molaires, puis renormalisation en fractions massiques       ---
        c_species = _draw(rng, _gen_distribution, 1.0, size=y_mole0.shape[0])
        y_mole_new = np.power(y_mole0, 1.0 + c_species / _gen_b)
        Y_unnorm = y_mole_new * _gen_W
        s = Y_unnorm.sum()
        if s <= 0:
            continue
        Y_new = Y_unnorm / s

        try:
            # reconstruction de l'énergie totale = énergie sensible perturbée
            # + énergie de formation de la NOUVELLE composition
            gas.TPY = _gen_T_ref, P0, Y_new
            e_ref_new = gas.enthalpy_mass if _gen_reactorType == 'cstP' else gas.int_energy_mass
            e_total_new = e_sens_new + e_ref_new

            if _gen_reactorType == 'cstP':
                gas.HPY = e_total_new, P0, Y_new
                P_new = P0
            else:  # 'cstV'
                gas.UVY = e_total_new, v0, Y_new
                P_new = gas.P
        except ct.CanteraError:
            continue
        T_new = gas.T

        # --- (3) acceptation / rejet (Liu : bornes élargies + tolérance  ---
        # --- relative sur les ratios élémentaires)                       ---
        if not (T_accept_range[0] <= T_new <= T_accept_range[1]):
            continue

        z_new = gas.mixture_fraction(_gen_fuel, _gen_oxidizer, basis='mass')
        if not (z_accept_range[0] <= z_new <= z_accept_range[1]):
            continue

        CH_new = _molar_ratio(gas, 'C', 'H')
        if abs(CH_new - CH0) > elem_ratio_tol * abs(CH0):
            continue

        ON_new = _molar_ratio(gas, 'O', 'N')
        if abs(ON_new - ON0) > elem_ratio_tol * abs(ON0):
            continue

        return np.array([T_new, P_new] + list(Y_new))

    return None  # état de base abandonné après max_attempts


# --------------------------------------------------------------------------- #
#  Fonction principale                                                       #
# --------------------------------------------------------------------------- #

def randomizedDatasetGeneration(
    X,
    n_target: int,
    kinetics: str,
    dt: float,
    reactorType: str,
    fuel: str = 'CH4:1',
    oxidizer: str = 'O2:0.23,N2:0.77',
    T_select_range: tuple = (500.0, 3000.0),   # Liu, Section 2.2
    z_select_range: tuple = (0.00, 0.36),      # Liu, Section 2.2 (leur "f")
    T_accept_range: tuple = (490.0, 3000.0),   # Liu, Section 2.2
    z_accept_range: tuple = (0.00, 0.39),      # Liu, Section 2.2
    elem_ratio_tol: float = 0.02,              # Liu : tolérance relative 2% (C/H, O/N molaires)
    a_coef: float = 8.0,                       # Ding, Eq. (5) -- non redéfini par Liu
    b_coef: float = 5.0,                       # Ding, Eq. (6) -- non redéfini par Liu
    max_attempts: int = 10,                    # Ding -- non redéfini par Liu
    T_ref_sensible: float = 298.15,            # !!! non précisé par Ding ni Liu
    distribution: str = 'uniform',             # 'uniform' (Ding/Liu) ou 'gaussian' (extension)*
    P_perturbation_mode: str | None = None,    # None, 'multiplicative' ou 'additive'
    P_perturbation_size: float = 0.0,          # multiplicatif : fraction (0.05 = ±5%) ; additif : Pa
    P_perturbation_distribution: str | None = None,  # None -> même loi que `distribution`
    minmax_sample_size: int | None = None,
    n_workers: int | None = None,
    chunksize: int = 50,
    seed: int | None = None,
):
    """
    Génère n_target nouveaux couples (X, Y) suivant la structure de Ding
    et al. (2021, Eqs. 5-7) raffinée par Liu et al. (2026, Section 2.2),
    puis calcule les labels Y par intégration Cantera.

    Paramètres
    ----------
    X : serie
        Base de données de référence, colonnes [T, P, Y_1, ..., Y_K].
    n_target : int
        Nombre d'états randomisés à générer.
    kinetics : str
        Chemin vers le mécanisme Cantera (YAML).
    dt : float
        Intervalle d'intégration pour le calcul des labels Y.
    reactorType : str
        'cstV' ou 'cstP' -- contrôle à la fois la nature de la perturbation
        énergétique (h_sensible en cstP, u_sensible en cstV) et le type de
        réacteur Cantera utilisé pour l'intégration.
    fuel, oxidizer : str
        Compositions Cantera utilisées pour la fraction de mélange
        (Bilger, via Cantera).
    T_select_range, z_select_range : tuple
        Bornes de sélection de l'état de base (Liu, Section 2.2).
    T_accept_range, z_accept_range : tuple
        Bornes d'acceptation de l'état perturbé, élargies par rapport à la
        sélection (Liu, Section 2.2).
    elem_ratio_tol : float
        Tolérance RELATIVE (2% dans Liu et al.) sur la dérive des ratios
        élémentaires molaires C/H et O/N par rapport à l'état de base.
    a_coef, b_coef : float
        Coefficients des Eqs. (5)-(6) de Ding et al., non redéfinis par
        Liu et al. : a=8, b=5.
    max_attempts : int
        Nombre de tentatives de perturbation par état de base avant
        abandon (10, valeur de Ding et al., non redéfinie par Liu).
    T_ref_sensible : float
        Température de référence (K) séparant énergie sensible et énergie
        de formation -- non précisée par les deux articles, 298.15 K par
        défaut.
    distribution : str
        Loi de tirage des variables aléatoires c (Eq. 5) et c_species
        (Eqs. 6-7) : 'uniform' (U(-1,1), valeur de Ding/Liu, défaut) ou
        'gaussian' (N(0,1), extension non spécifiée par les deux articles).
    P_perturbation_mode : None, 'multiplicative' ou 'additive'
        Perturbation supplémentaire de P appliquée sur l'échantillon accepté,
        APRES tout le calcul de randomisation (identique en cstP et cstV) et
        AVANT l'intégration. Émule les variations de pression extérieures
        dues au solveur CFD. None -> aucune perturbation.
    P_perturbation_size : float
        Amplitude de la perturbation. Multiplicatif : P' = P*(1 + size*c)
        (size sans unité, ex. 0.05 pour ±5 %). Additif : P' = P + size*c
        (size en Pa). c ~ U(-1,1) ou N(0,1) selon la loi choisie.
    P_perturbation_distribution : str ou None
        'uniform' ou 'gaussian'. None -> reprend `distribution`.
    minmax_sample_size : int ou None
        Sous-échantillonnage de X pour le calcul des bornes de l'énergie
        sensible (Eq. 5). None -> calcul sur X entière.
    n_workers, chunksize, seed :
        Paramètres de parallélisation / reproductibilité.

    Retour
    ------
    X_out, Y_out : deux objets `serie` de shape (n_target, nCols)
    """
    if distribution not in ('uniform', 'gaussian'):
        raise ValueError(f"distribution doit être 'uniform' ou 'gaussian', reçu '{distribution}'")

    if P_perturbation_mode not in (None, 'multiplicative', 'additive'):
        raise ValueError(
            f"P_perturbation_mode doit être None, 'multiplicative' ou 'additive', "
            f"reçu '{P_perturbation_mode}'"
        )
    if P_perturbation_distribution is None:
        P_perturbation_distribution = distribution
    if P_perturbation_distribution not in ('uniform', 'gaussian'):
        raise ValueError(
            f"P_perturbation_distribution doit être 'uniform' ou 'gaussian', "
            f"reçu '{P_perturbation_distribution}'"
        )

    col_names = X.colName
    states = X.data.cpu().detach().numpy()  # (N, nCols)
    N = states.shape[0]
    species_names = list(col_names[2:])

    if n_workers is None:
        n_workers = mp.cpu_count()

    rng_master = np.random.default_rng(seed)

    # --- bornes de l'énergie SENSIBLE (Eq. 5 raffinée par Liu) sur la    ---
    # --- base de référence (ou un sous-échantillon)                      ---
    if minmax_sample_size is not None and minmax_sample_size < N:
        idx_sample = rng_master.choice(N, size=minmax_sample_size, replace=False)
        sample_states = states[idx_sample]
    else:
        sample_states = states

    with Pool(processes=n_workers, initializer=_bounds_worker_init,
              initargs=(kinetics, reactorType, T_ref_sensible)) as pool:
        energies = list(tqdm(
            pool.imap(_bounds_worker_task, list(sample_states), chunksize=chunksize),
            total=len(sample_states),
            desc="Calcul de min/max de l'énergie sensible (Eq. 5)",
        ))
    sens_min = float(np.min(energies))
    sens_max = float(np.max(energies))
    energy_symbol = "h_sens" if reactorType == 'cstP' else "u_sens"
    print(f"[randomizedDatasetGeneration] {energy_symbol}_min = {sens_min:.6g} J/kg, "
          f"{energy_symbol}_max = {sens_max:.6g} J/kg  (Eq. 5, sur {len(sample_states)} états, "
          f"distribution='{distribution}')")

    # --- génération randomisée avec rejet ---
    accepted = []
    pbar = tqdm(total=n_target, desc="Randomisation (structure Ding + raffinements Liu)")
    with Pool(processes=n_workers, initializer=_generation_worker_init,
              initargs=(kinetics, fuel, oxidizer, reactorType, species_names,
                        sens_min, sens_max, a_coef, b_coef, T_ref_sensible, distribution)) as pool:
        while len(accepted) < n_target:
            n_missing = n_target - len(accepted)
            batch_n = max(n_workers * chunksize, n_missing * 2)

            idx = rng_master.integers(0, N, size=batch_n)  # tirage avec remise
            seeds = rng_master.integers(0, 2**31 - 1, size=batch_n)

            args_list = [
                (states[i], int(s), T_select_range, z_select_range,
                 T_accept_range, z_accept_range, elem_ratio_tol, max_attempts)
                for i, s in zip(idx, seeds)
            ]

            for res in pool.imap(_generation_worker_task, args_list, chunksize=chunksize):
                if res is not None:
                    accepted.append(res)
                    pbar.update(1)
                if len(accepted) >= n_target:
                    break
    pbar.close()

    X_new = np.array(accepted[:n_target])

    # --- perturbation supplémentaire de P (émulation du solveur CFD) ---
    # Appliquée après toute la randomisation, en cstP comme en cstV.
    # T et Y restent inchangés ; l'intégration part de (T, P', Y).
    if P_perturbation_mode is not None and P_perturbation_size != 0.0:
        X_new[:, 1] = _perturb_pressure(
            X_new[:, 1], rng_master, P_perturbation_mode,
            P_perturbation_size, P_perturbation_distribution,
        )
        print(f"[randomizedDatasetGeneration] Perturbation de P ({P_perturbation_mode}, "
              f"size={P_perturbation_size}, loi='{P_perturbation_distribution}') appliquée.")

    # --- intégration directe Cantera pour obtenir les labels Y ---
    with Pool(processes=n_workers, initializer=_integration_worker_init,
              initargs=(kinetics, reactorType, dt)) as pool:
        Y_list = list(tqdm(
            pool.imap(_integration_worker_task, list(X_new), chunksize=chunksize),
            total=n_target,
            desc="Intégration Cantera (labels Y)",
        ))
    Y_arr = np.array(Y_list)

    X_out = serie(X_new, col_names, storingDevice=X.storingDevice, storingDtype=X.storingDtype)
    Y_out = serie(Y_arr, col_names, storingDevice=X.storingDevice, storingDtype=X.storingDtype)

    return X_out, Y_out
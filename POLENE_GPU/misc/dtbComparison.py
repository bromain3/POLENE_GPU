import numpy as np
import matplotlib.pyplot as plt

try:
    import cantera as ct
except ImportError:
    ct = None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS PRIVÉS
# ══════════════════════════════════════════════════════════════════════════════

def _to_numpy(X) -> np.ndarray:
    """Accepte ndarray, tensor PyTorch, ou objet POLENE (`serie`) avec .data.detach()."""
    if isinstance(X, np.ndarray):
        return X
    try:
        return X.data.detach().cpu().numpy()
    except AttributeError:
        pass
    try:
        return X.numpy()
    except AttributeError:
        pass
    return np.asarray(X)


def _get_col_names(X, n_cols: int, feature_names=None) -> list:
    """
    Récupère les noms de colonnes depuis `feature_names`, ou depuis
    `X.colName` si `X` est un objet POLENE (`serie`), sinon retombe sur
    F0, F1, ...
    """
    if feature_names is not None:
        return list(feature_names)
    col_name = getattr(X, "colName", None)
    if col_name is not None:
        return list(col_name)
    return [f"F{i}" for i in range(n_cols)]


def _resolve_index(col, col_names: list) -> int:
    """Convertit un identifiant de colonne (int ou str) en index entier."""
    if isinstance(col, (int, np.integer)):
        return int(col)
    if col not in col_names:
        raise ValueError(f"Colonne '{col}' introuvable parmi {col_names}")
    return col_names.index(col)


def _is_species_col(idx: int, col_names: list, T_col, P_col) -> bool:
    """
    Une colonne est considérée comme une espèce (donc candidate à l'échelle
    log) si ce n'est ni la colonne T ni la colonne P.
    """
    iT = _resolve_index(T_col, col_names)
    iP = _resolve_index(P_col, col_names)
    return idx not in (iT, iP)


def _resolve_log_flag(flag, idx: int, col_names: list, T_col, P_col) -> bool:
    """
    Résout un paramètre log_x/log_y qui peut valoir True, False, ou "auto".
    En mode "auto" (défaut), l'échelle log est activée automatiquement pour
    toute colonne détectée comme une espèce (cf. `_is_species_col`), et
    désactivée pour T et P.
    """
    if flag == "auto":
        return _is_species_col(idx, col_names, T_col, P_col)
    return bool(flag)


def _clip_for_log(values: np.ndarray, use_log: bool, log_floor: float) -> np.ndarray:
    """
    Si `use_log` est vrai, remplace les valeurs <= 0 (fractions massiques
    nulles ou négatives numériquement) par `log_floor`, pour que l'échelle
    log matplotlib ne fasse pas disparaître ces points.
    """
    if not use_log:
        return values
    return np.clip(values, log_floor, None)


# ══════════════════════════════════════════════════════════════════════════════
# 0. SOUS-ÉCHANTILLONNAGE UNIFORME
# ══════════════════════════════════════════════════════════════════════════════

def undersampleUniform(X, n_max: int | None, seed: int | None = None) -> np.ndarray:
    """
    Sous-échantillonne uniformément (sans remise) une base de données, pour
    en faciliter le tracé lorsque le nombre de points est trop élevé.

    Parameters
    ----------
    X      : données brutes (N, F) — array, tensor, ou objet POLENE (`serie`)
    n_max  : nombre maximal de points conservés (None -> pas de sous-échantillonnage)
    seed   : graine du générateur aléatoire (reproductibilité)

    Retour
    ------
    np.ndarray, shape (min(N, n_max), F)
    """
    arr = _to_numpy(X)
    N = arr.shape[0]
    if n_max is None or N <= n_max:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n_max, replace=False)
    return arr[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRANCHE 2D feature_i × feature_j -- COMPARAISON X_ini / X_final
# ══════════════════════════════════════════════════════════════════════════════

def plotFeatureSlice(
    X_ini,
    X_final,
    feature_i,
    feature_j,
    feature_names: list | None = None,
    max_points: int | None = 20000,
    seed: int | None = None,
    labels: tuple = ("initiale", "finale"),
    colors: tuple = ("darkorange", "steelblue"),
    s: float = 3.0,
    alpha: float = 0.8,
    log_x="auto",
    log_y="auto",
    log_floor: float = 1e-30,
    T_col="T",
    P_col="P",
    ax=None,
    prints: bool = True,
):
    """
    Compare deux bases de données (ex: base initiale vs base augmentée) sur
    une tranche 2D feature_i × feature_j, les deux nuages étant superposés
    avec des couleurs différentes (comme la Fig. 1(a) de l'article).

    Parameters
    ----------
    X_ini, X_final : données brutes (N, F) — array, tensor, ou objet POLENE,
                      partageant la même structure de colonnes.
    feature_i, feature_j : int ou str
                      indices ou noms des colonnes à tracer en x et y.
    feature_names   : noms des colonnes (sinon déduits de X_ini.colName, ou F0,F1,...)
    max_points      : nb max de points affichés par base (sous-échantillonnage
                      uniforme via `undersampleUniform`)
    labels, colors  : légendes et couleurs de X_ini / X_final
    log_x, log_y    : True / False / "auto" (défaut). En mode "auto", l'échelle
                      log est activée automatiquement dès que la colonne
                      correspondante est une espèce (i.e. différente de
                      `T_col` et `P_col`), et désactivée pour T et P.
    log_floor       : plancher utilisé pour "sauver" les valeurs <= 0
                      (fractions massiques nulles) quand l'échelle log est
                      active, sinon elles disparaîtraient du graphe.
    T_col, P_col    : nom ou indice des colonnes T et P, utilisées pour la
                      détection automatique des espèces (mode "auto").
    ax              : axe matplotlib existant (sinon une nouvelle figure est créée)
    prints          : affiche le nombre de points effectivement tracés

    Retour
    ------
    ax : l'axe matplotlib utilisé
    """
    arr_ini_full = _to_numpy(X_ini)
    n_cols = arr_ini_full.shape[1]
    col_names = _get_col_names(X_ini, n_cols, feature_names)
    i = _resolve_index(feature_i, col_names)
    j = _resolve_index(feature_j, col_names)

    use_log_x = _resolve_log_flag(log_x, i, col_names, T_col, P_col)
    use_log_y = _resolve_log_flag(log_y, j, col_names, T_col, P_col)

    arr_ini = undersampleUniform(X_ini, max_points, seed)
    arr_fin = undersampleUniform(X_final, max_points, seed)

    if prints:
        print(f"Tracé de {arr_ini.shape[0]} points ({labels[0]}) "
              f"et {arr_fin.shape[0]} points ({labels[1]})"
              f" -- échelle log : x={use_log_x}, y={use_log_y}")

    x_ini = _clip_for_log(arr_ini[:, i], use_log_x, log_floor)
    y_ini = _clip_for_log(arr_ini[:, j], use_log_y, log_floor)
    x_fin = _clip_for_log(arr_fin[:, i], use_log_x, log_floor)
    y_fin = _clip_for_log(arr_fin[:, j], use_log_y, log_floor)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(x_fin, y_fin, s=s, alpha=alpha,
               color=colors[1], edgecolor="none", label=labels[1])
    ax.scatter(x_ini, y_ini, s=s, alpha=alpha,
               color=colors[0], edgecolor="none", label=labels[0])

    if use_log_x:
        ax.set_xscale("log")
    if use_log_y:
        ax.set_yscale("log")

    ax.set_xlabel(col_names[i])
    ax.set_ylabel(col_names[j])
    ax.set_title(f"{col_names[j]} vs {col_names[i]}")
    ax.legend(markerscale=3, framealpha=0.7)

    if created_fig:
        plt.tight_layout()
        plt.show()

    return ax


def plotMultipleFeatureSlices(
    X_ini,
    X_final,
    pairs: list,
    feature_names: list | None = None,
    max_points: int | None = 20000,
    seed: int | None = None,
    labels: tuple = ("initiale", "finale"),
    colors: tuple = ("darkorange", "steelblue"),
    s: float = 3.0,
    alpha: float = 0.8,
    log_x="auto",
    log_y="auto",
    log_floor: float = 1e-30,
    T_col="T",
    P_col="P",
    n_cols_grid: int = 3,
    figsize_per_ax: tuple = (4.5, 4.0),
):
    """
    Trace plusieurs tranches feature_i × feature_j sur une grille de
    sous-figures (le sous-échantillonnage est fait une seule fois et
    partagé entre tous les sous-plots, pour rester cohérent et rapide).

    Parameters
    ----------
    pairs : liste de tuples (feature_i, feature_j), int ou str
    log_x, log_y, log_floor, T_col, P_col : cf. `plotFeatureSlice` (même
        détection automatique de l'échelle log par colonne)
    autres paramètres : cf. `plotFeatureSlice`
    """
    n_plots = len(pairs)
    n_cols = min(n_cols_grid, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    arr_ini = undersampleUniform(X_ini, max_points, seed)
    arr_fin = undersampleUniform(X_final, max_points, seed)
    n_cols_data = arr_ini.shape[1]
    col_names = _get_col_names(X_ini, n_cols_data, feature_names)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_ax[0] * n_cols, figsize_per_ax[1] * n_rows),
    )
    axes_flat = np.atleast_1d(axes).flatten()

    for loc, (fi, fj) in enumerate(pairs):
        i = _resolve_index(fi, col_names)
        j = _resolve_index(fj, col_names)
        use_log_x = _resolve_log_flag(log_x, i, col_names, T_col, P_col)
        use_log_y = _resolve_log_flag(log_y, j, col_names, T_col, P_col)

        x_ini = _clip_for_log(arr_ini[:, i], use_log_x, log_floor)
        y_ini = _clip_for_log(arr_ini[:, j], use_log_y, log_floor)
        x_fin = _clip_for_log(arr_fin[:, i], use_log_x, log_floor)
        y_fin = _clip_for_log(arr_fin[:, j], use_log_y, log_floor)

        ax = axes_flat[loc]
        ax.scatter(x_fin, y_fin, s=s, alpha=alpha,
                   color=colors[1], edgecolor="none", label=labels[1])
        ax.scatter(x_ini, y_ini, s=s, alpha=alpha,
                   color=colors[0], edgecolor="none", label=labels[0])
        if use_log_x:
            ax.set_xscale("log")
        if use_log_y:
            ax.set_yscale("log")
        ax.set_xlabel(col_names[i], fontsize=9)
        ax.set_ylabel(col_names[j], fontsize=9)
        ax.tick_params(labelsize=7)
        if loc == 0:
            ax.legend(fontsize=7, markerscale=3, framealpha=0.7)

    for loc in range(n_plots, len(axes_flat)):
        axes_flat[loc].set_visible(False)

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 2. PLAN (T, f) -- COMPARAISON MANIFOLD / RANDOMISÉ (cf. Fig. 1 de l'article)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_T_f(arr, col_names, gas, fuel, oxidizer, T_col, P_col):
    """Calcule T et la fraction de mélange (Bilger, via Cantera) pour chaque ligne de `arr`."""
    iT = _resolve_index(T_col, col_names)
    iP = _resolve_index(P_col, col_names)
    Y_cols = [c for c in range(len(col_names)) if c not in (iT, iP)]

    T = arr[:, iT].astype(np.float64)
    f = np.empty(arr.shape[0], dtype=np.float64)
    for row in range(arr.shape[0]):
        gas.TPY = arr[row, iT], arr[row, iP], arr[row, Y_cols]
        f[row] = gas.mixture_fraction(fuel, oxidizer, basis="mass")
    return T, f


def plotTfPlane(
    X_ini,
    X_final,
    kinetics: str,
    fuel: str = "CH4:1",
    oxidizer: str = "O2:0.23,N2:0.77",
    T_col="T",
    P_col="P",
    feature_names: list | None = None,
    max_points: int | None = 5000,
    seed: int | None = None,
    labels: tuple = ("manifold", "randomisé"),
    colors: tuple = ("darkorange", "steelblue"),
    s: float = 3.0,
    alpha: float = 0.8,
    bins: int = 80,
    figsize: tuple = (13, 5),
    prints: bool = True,
):
    """
    Reproduit la Fig. 1 de l'article (Liu, Kronenburg & Zirwes, 2026) :
    (a) température vs fraction de mélange, (b) distribution de probabilité
    de la température, pour comparer une base "manifold" (X_ini) et une
    base "randomisée" (X_final).

    Le calcul de la fraction de mélange nécessite un appel Cantera par
    point (gas.mixture_fraction) : on sous-échantillonne donc d'abord
    (`max_points`) pour rester rapide sur de grandes bases -- augmenter ce
    nombre donne un plot plus fidèle mais plus lent.

    Parameters
    ----------
    X_ini, X_final : bases à comparer, mêmes colonnes [T, P, Y_1...Y_K]
    kinetics       : chemin vers le mécanisme Cantera (YAML), ex. gri30.yaml
    fuel, oxidizer : compositions Cantera pour la fraction de mélange de
                     Bilger (par défaut : CH4 pur / air massique 23% O2, 77% N2,
                     cf. Section 2.1 de l'article)
    T_col, P_col   : nom ou indice des colonnes T et P
    max_points     : nb max de points par base utilisés pour le calcul de f
    prints         : affiche le nombre de points utilisés

    Retour
    ------
    dict avec les arrays T_ini, f_ini, T_fin, f_fin utilisés pour le tracé
    """
    if ct is None:
        raise ImportError("Cantera n'est pas installé : `pip install cantera`.")

    gas = ct.Solution(kinetics)

    arr_ini_full = _to_numpy(X_ini)
    n_cols = arr_ini_full.shape[1]
    col_names = _get_col_names(X_ini, n_cols, feature_names)

    arr_ini = undersampleUniform(X_ini, max_points, seed)
    arr_fin = undersampleUniform(X_final, max_points, seed)

    if prints:
        print(f"Calcul de T,f sur {arr_ini.shape[0]} points ({labels[0]}) "
              f"et {arr_fin.shape[0]} points ({labels[1]})...")

    T_ini, f_ini = _compute_T_f(arr_ini, col_names, gas, fuel, oxidizer, T_col, P_col)
    T_fin, f_fin = _compute_T_f(arr_fin, col_names, gas, fuel, oxidizer, T_col, P_col)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # (a) T vs f -- équivalent Fig. 1(a)
    ax1.scatter(f_fin, T_fin, s=s, alpha=alpha, color=colors[1],
                edgecolor="none", label=labels[1])
    ax1.scatter(f_ini, T_ini, s=s, alpha=alpha, color=colors[0],
                edgecolor="none", label=labels[0])
    ax1.set_xlabel("Fraction de mélange (-)")
    ax1.set_ylabel("Température (K)")
    ax1.set_title("(a) T vs fraction de mélange")
    ax1.legend(markerscale=3, framealpha=0.7)

    # (b) histogramme (densité de probabilité) de T -- équivalent Fig. 1(b)
    ax2.hist(T_ini, bins=bins, density=True, color=colors[0],
             alpha=0.6, edgecolor="none", label=labels[0])
    ax2.hist(T_fin, bins=bins, density=True, color=colors[1],
             alpha=0.6, edgecolor="none", label=labels[1])
    ax2.set_xlabel("Température (K)")
    ax2.set_ylabel("Probabilité (-)")
    ax2.set_title("(b) distribution de T")
    ax2.legend(framealpha=0.7)

    plt.tight_layout()
    plt.show()

    return {"T_ini": T_ini, "f_ini": f_ini, "T_fin": T_fin, "f_fin": f_fin}


# ══════════════════════════════════════════════════════════════════════════════
# 3. ÉNERGIE TOTALE vs ÉNERGIE SENSIBLE -- COMPARAISON X_ini / X_final
# ══════════════════════════════════════════════════════════════════════════════

def _compute_total_and_sensible_energy(arr, col_names, gas, reactorType, T_ref, P_ref, T_col, P_col):
    """
    Calcule, pour chaque ligne de `arr`, l'énergie totale (h en cstP, u en
    cstV) et sa composante sensible (= valeur totale moins la valeur prise
    par le même potentiel à T_ref, à composition Y inchangée -- ce qui
    retranche l'enthalpie/énergie de formation, indépendante de T pour un
    gaz idéal).
    """
    iT = _resolve_index(T_col, col_names)
    iP = _resolve_index(P_col, col_names)
    Y_cols = [c for c in range(len(col_names)) if c not in (iT, iP)]

    total = np.empty(arr.shape[0], dtype=np.float64)
    sensible = np.empty(arr.shape[0], dtype=np.float64)

    for row in range(arr.shape[0]):
        T = arr[row, iT]
        P = arr[row, iP]
        Y = arr[row, Y_cols]

        gas.TPY = T, P, Y
        e_tot = gas.enthalpy_mass if reactorType == "cstP" else gas.int_energy_mass

        # h et u d'un gaz idéal ne dépendent que de (T, Y), pas de P/V --
        # on peut donc réutiliser la même pression pour l'état de référence
        gas.TPY = T_ref, P, Y
        e_ref = gas.enthalpy_mass if reactorType == "cstP" else gas.int_energy_mass

        total[row] = e_tot
        sensible[row] = e_tot - e_ref

    return total, sensible


def plotEnergyComparison(
        X_ini,
        X_final,
        kinetics: str,
        reactorType: str,
        T_ref: float = 298.15,
        T_col="T",
        P_col="P",
        feature_names: list | None = None,
        max_points: int | None = 20000,
        seed: int | None = None,
        labels: tuple = ("initiale", "finale"),
        colors: tuple = ("darkorange", "steelblue"),
        bins: int = 80,
        figsize: tuple = (13, 5),
        prints: bool = True,
):
    """
    Compare deux bases de données (ex: base initiale vs base augmentée) sur
    leur potentiel énergétique thermodynamique : (a) énergie totale
    (enthalpie massique h en 'cstP', énergie interne massique u en 'cstV'),
    (b) énergie sensible correspondante, c'est-à-dire l'énergie totale à
    laquelle on retranche l'énergie de formation (valeur du même potentiel
    évaluée à T_ref, à composition inchangée). Utile notamment pour
    vérifier la perturbation de l'Eq. (5) de Ding et al. (2021) /
    l'extension cstV de `randomizedDatasetGeneration.py`.

    Parameters
    ----------
    X_ini, X_final : bases à comparer, mêmes colonnes [T, P, Y_1...Y_K]
    kinetics       : chemin vers le mécanisme Cantera (YAML)
    reactorType    : 'cstP' (énergie tracée = enthalpie h) ou 'cstV'
                     (énergie tracée = énergie interne u) -- doit
                     correspondre au reactorType utilisé pour générer
                     X_final si on veut comparer une perturbation h/u.
    T_ref          : température de référence (K) utilisée pour soustraire
                     l'énergie de formation et obtenir l'énergie sensible
                     (298.15 K par défaut)
    T_col, P_col   : nom ou indice des colonnes T et P
    max_points     : nb max de points par base utilisés pour le calcul
                     (un appel Cantera par point -> coûteux sur de grandes
                     bases, d'où le sous-échantillonnage)
    prints         : affiche le nombre de points utilisés

    Retour
    ------
    dict avec les arrays total_ini, sensible_ini, total_fin, sensible_fin
    """
    if ct is None:
        raise ImportError("Cantera n'est pas installé : `pip install cantera`.")
    if reactorType not in ("cstP", "cstV"):
        raise ValueError(f"reactorType doit être 'cstP' ou 'cstV', reçu '{reactorType}'")

    gas = ct.Solution(kinetics)

    arr_ini_full = _to_numpy(X_ini)
    n_cols = arr_ini_full.shape[1]
    col_names = _get_col_names(X_ini, n_cols, feature_names)

    arr_ini = undersampleUniform(X_ini, max_points, seed)
    arr_fin = undersampleUniform(X_final, max_points, seed)

    energy_symbol = "h" if reactorType == "cstP" else "u"

    if prints:
        print(f"Calcul de l'énergie ({energy_symbol}) sur {arr_ini.shape[0]} points "
              f"({labels[0]}) et {arr_fin.shape[0]} points ({labels[1]})...")

    # même pression de référence P_ref pas nécessaire : h et u d'un gaz
    # idéal ne dépendent que de (T, Y), pas de P -- cf. `_compute_total_and_sensible_energy`
    tot_ini, sens_ini = _compute_total_and_sensible_energy(
        arr_ini, col_names, gas, reactorType, T_ref, None, T_col, P_col)
    tot_fin, sens_fin = _compute_total_and_sensible_energy(
        arr_fin, col_names, gas, reactorType, T_ref, None, T_col, P_col)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # (a) énergie totale
    ax1.hist(tot_ini, bins=bins, density=True, color=colors[0],
             alpha=0.6, edgecolor="none", label=labels[0])
    ax1.hist(tot_fin, bins=bins, density=True, color=colors[1],
             alpha=0.6, edgecolor="none", label=labels[1])
    ax1.set_xlabel(f"{energy_symbol} total (J/kg)")
    ax1.set_ylabel("Densité (-)")
    ax1.set_title(f"(a) énergie totale ({energy_symbol})")
    ax1.legend(framealpha=0.7)

    # (b) énergie sensible
    ax2.hist(sens_ini, bins=bins, density=True, color=colors[0],
             alpha=0.6, edgecolor="none", label=labels[0])
    ax2.hist(sens_fin, bins=bins, density=True, color=colors[1],
             alpha=0.6, edgecolor="none", label=labels[1])
    ax2.set_xlabel(f"{energy_symbol} sensible (J/kg), réf. {T_ref:.1f} K")
    ax2.set_ylabel("Densité (-)")
    ax2.set_title(f"(b) énergie sensible ({energy_symbol})")
    ax2.legend(framealpha=0.7)

    plt.tight_layout()
    plt.show()

    return {"total_ini": tot_ini, "sensible_ini": sens_ini,
            "total_fin": tot_fin, "sensible_fin": sens_fin}

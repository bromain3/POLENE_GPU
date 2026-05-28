import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS PRIVÉS
# ══════════════════════════════════════════════════════════════════════════════

def _to_numpy(X) -> np.ndarray:
    """Accepte ndarray, tensor PyTorch, ou objet POLENE avec .data.detach()."""
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


def _fit_and_project(X, n_components: int, scaler=None):
    """Fit scaler + PCA, retourne (pca, Z)."""
    X_scaled = _to_numpy(scaler.fit_transform(X))
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X_scaled)
    return pca, Z, scaler


def _pc_label(d: int, evr: np.ndarray | None) -> str:
    base = f"PC{d + 1}"
    if evr is not None and d < len(evr):
        return f"{base}  ({evr[d] * 100:.1f}%)"
    return base


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOADINGS PAR COMPOSANTE
# ══════════════════════════════════════════════════════════════════════════════

def plotLoadings(
    X,
    n_components: int,
    feature_names: list[str] | None = None,
    scaler=None,
    n_per_fig: int = 3,
    prints: bool = True,
):
    """
    Affiche les loadings de toutes les composantes PCA, par blocs de `n_per_fig`.

    Parameters
    ----------
    X             : données brutes (N, F) — array, tensor, ou objet POLENE
    n_components  : nombre de composantes PCA à calculer
    feature_names : noms des features (sinon "F0", "F1", …)
    scaler        : scaler sklearn-compatible (défaut : StandardScaler)
    n_per_fig     : composantes affichées par figure
    prints        : affiche la variance expliquée dans le terminal
    """
    pca, _, _ = _fit_and_project(X, n_components, scaler)
    evr = pca.explained_variance_ratio_

    if prints:
        print(f"Variance expliquée par {n_components} composantes : {evr.sum() * 100:.1f}%")

    n_feat = pca.components_.shape[1]
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(n_feat)]

    loadings = pca.components_ * np.sqrt(pca.explained_variance_[:, np.newaxis])
    fig_w = max(12.0, n_feat * 0.4)

    for start in range(0, n_components, n_per_fig):
        end = min(start + n_per_fig, n_components)
        n_cur = end - start

        fig, axes = plt.subplots(n_cur, 1, figsize=(fig_w, 4 * n_cur))
        fig.suptitle(f"PCA loadings — composantes {start + 1} à {end}", fontsize=14)
        if n_cur == 1:
            axes = [axes]

        for loc, d in enumerate(range(start, end)):
            order = np.argsort(np.abs(loadings[d]))[::-1]
            colors = ["tomato" if v >= 0 else "steelblue" for v in loadings[d][order]]
            axes[loc].bar(np.arange(n_feat), loadings[d][order], color=colors, edgecolor="none")
            axes[loc].set_xticks(np.arange(n_feat))
            axes[loc].set_xticklabels([feature_names[i] for i in order], rotation=90, fontsize=8)
            axes[loc].axhline(0, color="k", lw=0.8)
            axes[loc].set_title(f"PC{d + 1}  ({evr[d] * 100:.1f}% variance expliquée)")
            axes[loc].set_ylabel("loading")

        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 2. COUPE 2D  PCi × PCj
# ══════════════════════════════════════════════════════════════════════════════

def plot2DSlice(
    X,
    i: int = 0,
    j: int = 1,
    n_components: int | None = None,
    c=None,
    scaler=None,
    cmap: str = "viridis",
    s: float = 3.0,
    log_color: bool = False,
    prints: bool = True,
):
    """
    Scatter 2D de la coupe PCi × PCj.

    Parameters
    ----------
    X            : données brutes (N, F)
    i, j         : indices des composantes à afficher (0-based)
    n_components : nombre de composantes PCA (défaut : max(i,j)+1)
    c            : valeurs pour la couleur (N,) — None → couleur uniforme
    scaler       : scaler sklearn-compatible (défaut : StandardScaler)
    cmap         : colormap matplotlib
    s            : taille des points
    log_color    : normalisation log de la couleur
    prints       : affiche la variance expliquée
    """
    if n_components is None:
        n_components = max(i, j) + 1

    pca, Z, _ = _fit_and_project(X, n_components, scaler)
    evr = pca.explained_variance_ratio_

    if prints:
        print(f"Variance expliquée par {n_components} composantes : {evr.sum() * 100:.1f}%")

    xl = _pc_label(i, evr)
    yl = _pc_label(j, evr)
    norm = mcolors.LogNorm() if (log_color and c is not None) else None

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Z[:, i], Z[:, j], c=c, cmap=cmap, s=s, norm=norm, alpha=0.7)
    if c is not None:
        plt.colorbar(sc, ax=ax)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f"Projection {xl} × {yl}")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIGURE 3D (PCi, PCj, PCk) + 2D (PCi, PCj) côte à côte
# ══════════════════════════════════════════════════════════════════════════════

def plot3DSlice(
        X,
        i: int = 0,
        j: int = 1,
        k: int = 2,
        n_components: int | None = None,
        c=None,
        scaler=None,
        cmap: str = "viridis",
        s: float = 3.0,
        log_color: bool = False,
        elev: float = 25.0,
        azim: float = 45.0,
        figsize: tuple = (14, 6),
        max_points: int | None = 50000,
        prints: bool = True,
):
    """
    Figure avec à gauche un scatter 3D (PCi, PCj, PCk) et à droite un scatter 2D (PCi, PCj).

    Parameters
    ----------
    X            : données brutes (N, F)
    i, j, k      : indices des composantes (0-based)
    n_components : nombre de composantes PCA (défaut : max(i,j,k)+1)
    c            : couleur (N,) — None → couleur uniforme
    scaler       : scaler sklearn-compatible (défaut : StandardScaler)
    elev, azim   : angle de vue du scatter 3D
    max_points   : nb de points affichés (tirage uniforme sans remise) — None → tous
    prints       : affiche la variance expliquée
    """
    if n_components is None:
        n_components = max(i, j, k) + 1

    pca, Z, _ = _fit_and_project(X, n_components, scaler)
    evr = pca.explained_variance_ratio_

    if prints:
        print(f"Variance expliquée par {n_components} composantes : {evr.sum() * 100:.1f}%")

    # Sous-échantillonnage uniforme pour le plot uniquement
    c_np = np.asarray(c) if c is not None else None
    if max_points is not None and len(Z) > max_points:
        idx = np.random.default_rng().choice(len(Z), size=max_points, replace=False)
        Z_plot = Z[idx]
        c_plot = c_np[idx] if c_np is not None else None
        if prints:
            print(f"Affichage : {max_points} points sur {len(Z)} ({max_points / len(Z) * 100:.1f}%)")
    else:
        Z_plot = Z
        c_plot = c_np

    xl = _pc_label(i, evr)
    yl = _pc_label(j, evr)
    zl = _pc_label(k, evr)
    norm = mcolors.LogNorm() if (log_color and c_plot is not None) else None

    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"PCA — PC{i + 1}, PC{j + 1}, PC{k + 1}", fontsize=13)

    # 3D
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    sc3 = ax3.scatter(Z_plot[:, i], Z_plot[:, j], Z_plot[:, k], c=c_plot, cmap=cmap, s=s, norm=norm, alpha=0.7)
    ax3.set_xlabel(xl, labelpad=6)
    ax3.set_ylabel(yl, labelpad=6)
    ax3.set_zlabel(zl, labelpad=6)
    ax3.set_title(f"3D — PC{i + 1}, PC{j + 1}, PC{k + 1}")
    ax3.view_init(elev=elev, azim=azim)
    if c is not None:
        fig.colorbar(sc3, ax=ax3, shrink=0.5, pad=0.1)

    # 2D
    ax2 = fig.add_subplot(1, 2, 2)
    sc2 = ax2.scatter(Z_plot[:, i], Z_plot[:, j], c=c_plot, cmap=cmap, s=s, norm=norm, alpha=0.7)
    ax2.set_xlabel(xl)
    ax2.set_ylabel(yl)
    ax2.set_title(f"2D — PC{i + 1} × PC{j + 1}")
    if c is not None:
        fig.colorbar(sc2, ax=ax2)

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4. HISTOGRAMMES DE TOUTES LES PC (blocs de 9)
# ══════════════════════════════════════════════════════════════════════════════

def plotPChistogram(
    X,
    n_components: int,
    X_after=None,
    scaler=None,
    bins: int = 50,
    n_per_fig: int = 9,
    label_before: str = "avant",
    label_after: str = "après",
    color_before: str = "steelblue",
    color_after: str = "seagreen",
    figsize_per_ax: tuple = (4.5, 3.0),
    prints: bool = True,
):
    """
    Histogrammes de toutes les composantes PCA, par blocs de `n_per_fig` (grille 3×3).
    Si `X_after` est fourni, superpose avant/après (ex: avant/après resampling).

    Parameters
    ----------
    X            : données brutes (N, F)
    n_components : nombre de composantes PCA
    X_after      : données après transformation (M, F) — même espace de features que X
    scaler       : scaler sklearn-compatible (défaut : StandardScaler)
    bins         : bins de l'histogramme
    n_per_fig    : composantes par figure (9 → grille 3×3)
    prints       : affiche la variance expliquée
    """
    pca, Z, fitted_scaler = _fit_and_project(X, n_components, scaler)
    evr = pca.explained_variance_ratio_

    if prints:
        print(f"Variance expliquée par {n_components} composantes : {evr.sum() * 100:.1f}%")

    # Projette X_after avec le même pca/scaler fitté sur X
    Z_after = None
    if X_after is not None:
        X_after_np = _to_numpy(X_after)
        Z_after = pca.transform(_to_numpy(fitted_scaler.transform(X_after_np)))

    n_cols = int(np.ceil(np.sqrt(n_per_fig)))
    n_rows = int(np.ceil(n_per_fig / n_cols))
    fig_w = figsize_per_ax[0] * n_cols
    fig_h = figsize_per_ax[1] * n_rows

    for start in range(0, n_components, n_per_fig):
        end = min(start + n_per_fig, n_components)
        n_cur = end - start

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
        suptitle = f"Distributions PC — composantes {start + 1} à {end}"
        if Z_after is not None:
            suptitle += f"  [{label_before} vs {label_after}]"
        fig.suptitle(suptitle, fontsize=13, y=1.01)

        axes_flat = np.array(axes).flatten()

        for loc, d in enumerate(range(start, end)):
            ax = axes_flat[loc]
            label = _pc_label(d, evr)
            alpha = 0.65 if Z_after is not None else 0.85

            ax.hist(Z[:, d], bins=bins, density=True,
                    color=color_before, alpha=alpha, edgecolor="none", label=label_before)

            if Z_after is not None and d < Z_after.shape[1]:
                ax.hist(Z_after[:, d], bins=bins, density=True,
                        color=color_after, alpha=0.65, edgecolor="none", label=label_after)
                ax.legend(fontsize=7, framealpha=0.6)

            ax.set_title(label, fontsize=9)
            ax.set_xlabel(f"PC{d + 1}", fontsize=8)
            ax.set_ylabel("densité", fontsize=8)
            ax.tick_params(labelsize=7)

        for loc in range(n_cur, len(axes_flat)):
            axes_flat[loc].set_visible(False)

        plt.tight_layout()
        plt.show()

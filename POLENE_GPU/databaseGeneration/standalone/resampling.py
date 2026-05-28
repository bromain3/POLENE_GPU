from POLENE_GPU.preProcessing.scaler.stdScaler import stdScaler
from POLENE_GPU.misc.timer import Timer

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np



def resampleManifold(Xdtb, Ydtb, sampleNumbers, manifoldDim=2, nbins=50, plots=True, prints=True, scalerType=None):
    # ──────────────────────────────────────────────────────────────────────────────
    # VARIABLE DE PROGRÈS — PCA k-dimensionnel
    # ──────────────────────────────────────────────────────────────────────────────
    K_PCA = manifoldDim

    if scalerType is None:
        scaler_prog_var = stdScaler()
    else:
        scaler_prog_var = scalerType
    Xdtb_scaled = scaler_prog_var.fit_transform(Xdtb)

    pca = PCA(n_components=K_PCA)
    Z = pca.fit_transform(Xdtb_scaled.data.detach().cpu().numpy())  # (N, K_PCA)

    if prints:
        print(f"Variance expliquée par {K_PCA} composantes : {pca.explained_variance_ratio_.sum() * 100:.1f}%")
        print(f"  par composante : {pca.explained_variance_ratio_}")

    # ──────────────────────────────────────────────────────────────────────────────
    # IMPORTANCE DES FEATURES DANS LA PCA
    # ──────────────────────────────────────────────────────────────────────────────
    if plots:
        feature_names = Xdtb.colName

        components = pca.components_
        loadings = components * np.sqrt(pca.explained_variance_[:, np.newaxis])
        # ==========================================
        # affichage par blocs de 3 composantes max
        # ==========================================
        n_per_fig = 3
        for start_idx in range(0, K_PCA, n_per_fig):
            end_idx = min(start_idx + n_per_fig, K_PCA)
            n_current = end_idx - start_idx
            fig, axes = plt.subplots(n_current,1, figsize=(max(12, len(feature_names) * 0.4), 4 * n_current))
            fig.suptitle(f'PCA loadings — composantes {start_idx + 1} à {end_idx}', fontsize=14)
            # Cas où une seule subplot
            if n_current == 1:
                axes = [axes]

            for local_idx, d in enumerate(range(start_idx, end_idx)):
                order = np.argsort(np.abs(loadings[d]))[::-1]
                axes[local_idx].bar(np.arange(len(feature_names)), loadings[d][order], color=['tomato' if v >= 0 else 'steelblue' for v in loadings[d][order]], edgecolor='none')
                axes[local_idx].set_xticks(np.arange(len(feature_names)))
                axes[local_idx].set_xticklabels([feature_names[i] for i in order], rotation=90, fontsize=8)
                axes[local_idx].axhline(0, color='k', lw=0.8)
                axes[local_idx].set_title(f'PC{d + 1} ({pca.explained_variance_ratio_[d] * 100:.1f}% variance expliquée)')
                axes[local_idx].set_ylabel('loading')

            plt.tight_layout()
            plt.show()

    # ──────────────────────────────────────────────────────────────────────────────
    # POIDS PAR HISTOGRAMME k-D
    # ──────────────────────────────────────────────────────────────────────────────
    def compute_histogram_weights_kD(Z, n_bins):
        """
        Resampling uniforme sur le manifold via histogramme k-dimensionnel.

        - Z       : (N, k) coordonnées PCA
        - n_bins  : bins par dimension (attention : k=3,n_bins=20 → 8000 cellules)

        Retourne weights (N,) normalisés à 1.
        """
        N, k = Z.shape

        # Normalise chaque dim dans [0, 1] pour simplifier la gestion des bords
        Z_min = Z.min(axis=0)
        Z_max = Z.max(axis=0)
        Z_norm = (Z - Z_min) / (Z_max - Z_min + 1e-10)

        # Histogramme k-D
        edges = [np.linspace(0, 1, n_bins + 1) for _ in range(k)]
        counts, _ = np.histogramdd(Z_norm, bins=edges)  # (n_bins,) * k

        # Indice de bin pour chaque sample
        bin_indices = np.zeros((N, k), dtype=int)
        for d in range(k):
            idx_d = np.digitize(Z_norm[:, d], edges[d][:-1]) - 1
            bin_indices[:, d] = np.clip(idx_d, 0, n_bins - 1)

        # Poids = 1 / count de la cellule
        cell_counts = counts[tuple(bin_indices[:, d] for d in range(k))]  # (N,)
        weights = 1.0 / (cell_counts + 1e-10)
        weights /= weights.sum()

        # Stats sur l'occupation des cellules
        n_cells_total = n_bins ** k
        n_cells_occupied = np.sum(counts > 0)
        if prints:
            print(f"Cellules occupées : {n_cells_occupied} / {n_cells_total} ({100 * n_cells_occupied / n_cells_total:.1f}%)")
            print(f"  counts min/median/max dans cellules occupées : "
                  f"{counts[counts > 0].min():.0f} / "
                  f"{np.median(counts[counts > 0]):.0f} / "
                  f"{counts[counts > 0].max():.0f}")

        return weights, counts, bin_indices


    N_BINS_PER_DIM = nbins  # ← à adapter selon K_PCA (k=1→500, k=2→50, k=3→20, k=4→12)

    weights, counts, bin_indices = compute_histogram_weights_kD(Z, n_bins=N_BINS_PER_DIM)

    # ──────────────────────────────────────────────────────────────────────────────
    # VISUALISATION
    # ──────────────────────────────────────────────────────────────────────────────
    #TODO: ajouter une option pour resampler par rapport à des PC données care toutes les PC se resmblent plus u moins sauf les PC liées aux produits azotés
    if plots: #TODO: changer pour plot la distribution de toutes les PC avant et après resamplling, je pense que les PC qui ont des gros loadings en N2 seront toujours désequilibrées alors que les autres seront globalement bien foutues, à voir
        n_plots = min(K_PCA, 3)
        fig, axes = plt.subplots(2, n_plots, figsize=(6 * n_plots, 8))
        fig.suptitle(f'Diagnostic resampling — PCA {K_PCA}D, {N_BINS_PER_DIM} bins/dim')

        # Normalise axes en (2, n_plots) dans tous les cas
        if n_plots == 1:
            axes = axes[:, np.newaxis]  # (2,) → (2, 1)

        for d in range(n_plots):
            z_d = Z[:, d]
            # Histogramme avant
            axes[0, d].hist(z_d, bins=50, color='steelblue', edgecolor='none', density=True)
            axes[0, d].set_title(f'PC{d + 1} avant  ({pca.explained_variance_ratio_[d] * 100:.1f}% var)')
            axes[0, d].set_xlabel(f'PC{d + 1}')
            axes[0, d].set_ylabel('densité')

        # Distribution des poids
        w_plot = axes[1, 0]
        w_plot.hist(weights / weights.max(), bins=100, color='steelblue', edgecolor='none')
        w_plot.set_xlabel('poids normalisé')
        w_plot.set_ylabel('count')
        w_plot.set_title('Distribution des poids')

        # Poids vs PC1
        if n_plots > 1:
            axes[1, 1].scatter(Z[:, 0], Z[:, 1], c=weights, s=0.5,
                               cmap='hot', norm=plt.matplotlib.colors.LogNorm())
            axes[1, 1].set_xlabel('PC1');
            axes[1, 1].set_ylabel('PC2')
            axes[1, 1].set_title('Poids (log) dans PC1-PC2')

        if n_plots > 2:
            axes[1, 2].scatter(Z[:, 1], Z[:, 2], c=weights, s=0.5,
                               cmap='hot', norm=plt.matplotlib.colors.LogNorm())
            axes[1, 2].set_xlabel('PC2');
            axes[1, 2].set_ylabel('PC3')
            axes[1, 2].set_title('Poids (log) dans PC2-PC3')

        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────────────────────────
    # RESAMPLING
    # ──────────────────────────────────────────────────────────────────────────────
    rng = np.random.default_rng()
    idx = rng.choice(len(weights), size=sampleNumbers, replace=True, p=weights)

    X_new = Xdtb[idx]
    Y_new = Ydtb[idx]

    # ──────────────────────────────────────────────────────────────────────────────
    # VISUALISATION APRÈS RESAMPLING — projections PCA
    # ──────────────────────────────────────────────────────────────────────────────
    if plots:
        Z_new = pca.transform(scaler_prog_var.transform(X_new).data.detach().cpu().numpy())

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
        fig.suptitle(f'PC distributions avant / après resampling')
        if n_plots == 1:
            axes = [axes]

        for d in range(n_plots):
            axes[d].hist(Z[:, d], bins=50, alpha=0.6, density=True,
                         color='steelblue', label='avant', edgecolor='none')
            axes[d].hist(Z_new[:, d], bins=50, alpha=0.6, density=True,
                         color='seagreen', label='après', edgecolor='none')
            axes[d].set_title(f'PC{d + 1}')
            axes[d].set_xlabel(f'PC{d + 1}')
            axes[d].set_ylabel('densité')
            axes[d].legend()

        plt.tight_layout()
        plt.show()

    return(X_new, Y_new)


def resampleManifoldGivenPC(Xdtb, Ydtb, sampleNumbers, manifoldComponents, nbins=50, plots=True, prints=True, scalerType=None):
    # ──────────────────────────────────────────────────────────────────────────────
    # VARIABLE DE PROGRÈS — PCA k-dimensionnel
    # ──────────────────────────────────────────────────────────────────────────────
    components_0indexed = [c - 1 for c in manifoldComponents]  # passage en 0-indexé
    K_PCA = max(components_0indexed) + 1  # on calcule jusqu'à la composante max demandée
    K_resample = len(components_0indexed)

    if scalerType is None:
        scaler_prog_var = stdScaler()
    else:
        scaler_prog_var = scalerType
    Xdtb_scaled = scaler_prog_var.fit_transform(Xdtb)

    pca = PCA(n_components=K_PCA)
    Z_full = pca.fit_transform(Xdtb_scaled.data.detach().cpu().numpy())  # (N, K_PCA)
    Z = Z_full[:, components_0indexed]  # (N, K_resample) — composantes sélectionnées

    if prints:
        selected_labels = [f"PC{c+1}" for c in components_0indexed]
        var_selected = pca.explained_variance_ratio_[components_0indexed].sum()
        print(f"Composantes sélectionnées pour le resampling : {selected_labels}")
        print(f"Variance expliquée par ces composantes : {var_selected * 100:.1f}%")
        print(f"Variance totale expliquée par les {K_PCA} composantes calculées : {pca.explained_variance_ratio_.sum() * 100:.1f}%")
        print(f"  par composante : {pca.explained_variance_ratio_}")

    # ──────────────────────────────────────────────────────────────────────────────
    # IMPORTANCE DES FEATURES DANS LA PCA
    # ──────────────────────────────────────────────────────────────────────────────
    if plots:
        feature_names = Xdtb.colName
        components = pca.components_
        loadings = components * np.sqrt(pca.explained_variance_[:, np.newaxis])

        n_per_fig = 3
        for start_idx in range(0, K_resample, n_per_fig):
            end_idx = min(start_idx + n_per_fig, K_resample)
            n_current = end_idx - start_idx
            fig, axes = plt.subplots(n_current, 1, figsize=(max(12, len(feature_names) * 0.4), 4 * n_current))
            fig.suptitle(f'PCA loadings — composantes {[f"PC{c+1}" for c in components_0indexed[start_idx:end_idx]]}', fontsize=14)
            if n_current == 1:
                axes = [axes]

            for local_idx, d in enumerate(components_0indexed[start_idx:end_idx]):
                order = np.argsort(np.abs(loadings[d]))[::-1]
                axes[local_idx].bar(np.arange(len(feature_names)), loadings[d][order], color=['tomato' if v >= 0 else 'steelblue' for v in loadings[d][order]], edgecolor='none')
                axes[local_idx].set_xticks(np.arange(len(feature_names)))
                axes[local_idx].set_xticklabels([feature_names[i] for i in order], rotation=90, fontsize=8)
                axes[local_idx].axhline(0, color='k', lw=0.8)
                axes[local_idx].set_title(f'PC{d + 1} ({pca.explained_variance_ratio_[d] * 100:.1f}% variance expliquée)')
                axes[local_idx].set_ylabel('loading')

            plt.tight_layout()
            plt.show()

    # ──────────────────────────────────────────────────────────────────────────────
    # POIDS PAR HISTOGRAMME k-D
    # ──────────────────────────────────────────────────────────────────────────────
    def compute_histogram_weights_kD(Z, n_bins):
        N, k = Z.shape

        Z_min = Z.min(axis=0)
        Z_max = Z.max(axis=0)
        Z_norm = (Z - Z_min) / (Z_max - Z_min + 1e-10)

        edges = [np.linspace(0, 1, n_bins + 1) for _ in range(k)]
        counts, _ = np.histogramdd(Z_norm, bins=edges)

        bin_indices = np.zeros((N, k), dtype=int)
        for d in range(k):
            idx_d = np.digitize(Z_norm[:, d], edges[d][:-1]) - 1
            bin_indices[:, d] = np.clip(idx_d, 0, n_bins - 1)

        cell_counts = counts[tuple(bin_indices[:, d] for d in range(k))]
        weights = 1.0 / (cell_counts + 1e-10)
        weights /= weights.sum()

        n_cells_total = n_bins ** k
        n_cells_occupied = np.sum(counts > 0)
        if prints:
            print(f"Cellules occupées : {n_cells_occupied} / {n_cells_total} ({100 * n_cells_occupied / n_cells_total:.1f}%)")
            print(f"  counts min/median/max dans cellules occupées : "
                  f"{counts[counts > 0].min():.0f} / "
                  f"{np.median(counts[counts > 0]):.0f} / "
                  f"{counts[counts > 0].max():.0f}")

        return weights, counts, bin_indices

    weights, counts, bin_indices = compute_histogram_weights_kD(Z, n_bins=nbins)

    # ──────────────────────────────────────────────────────────────────────────────
    # VISUALISATION
    # ──────────────────────────────────────────────────────────────────────────────
    if plots:
        n_plots = min(K_resample, 3)
        fig, axes = plt.subplots(2, n_plots, figsize=(6 * n_plots, 8))
        fig.suptitle(f'Diagnostic resampling — composantes {[f"PC{c+1}" for c in components_0indexed]}, {nbins} bins/dim')

        if n_plots == 1:
            axes = axes[:, np.newaxis]

        for local_idx in range(n_plots):
            d = components_0indexed[local_idx]
            axes[0, local_idx].hist(Z_full[:, d], bins=50, color='steelblue', edgecolor='none', density=True)
            axes[0, local_idx].set_title(f'PC{d+1} avant  ({pca.explained_variance_ratio_[d] * 100:.1f}% var)')
            axes[0, local_idx].set_xlabel(f'PC{d+1}')
            axes[0, local_idx].set_ylabel('densité')

        w_plot = axes[1, 0]
        w_plot.hist(weights / weights.max(), bins=100, color='steelblue', edgecolor='none')
        w_plot.set_xlabel('poids normalisé')
        w_plot.set_ylabel('count')
        w_plot.set_title('Distribution des poids')

        if n_plots > 1:
            d0, d1 = components_0indexed[0], components_0indexed[1]
            axes[1, 1].scatter(Z_full[:, d0], Z_full[:, d1], c=weights, s=0.5,
                               cmap='hot', norm=plt.matplotlib.colors.LogNorm())
            axes[1, 1].set_xlabel(f'PC{d0+1}')
            axes[1, 1].set_ylabel(f'PC{d1+1}')
            axes[1, 1].set_title(f'Poids (log) dans PC{d0+1}-PC{d1+1}')

        if n_plots > 2:
            d1, d2 = components_0indexed[1], components_0indexed[2]
            axes[1, 2].scatter(Z_full[:, d1], Z_full[:, d2], c=weights, s=0.5,
                               cmap='hot', norm=plt.matplotlib.colors.LogNorm())
            axes[1, 2].set_xlabel(f'PC{d1+1}')
            axes[1, 2].set_ylabel(f'PC{d2+1}')
            axes[1, 2].set_title(f'Poids (log) dans PC{d1+1}-PC{d2+1}')

        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────────────────────────
    # RESAMPLING
    # ──────────────────────────────────────────────────────────────────────────────
    rng = np.random.default_rng()
    idx = rng.choice(len(weights), size=sampleNumbers, replace=True, p=weights)

    X_new = Xdtb[idx]
    Y_new = Ydtb[idx]

    # ──────────────────────────────────────────────────────────────────────────────
    # VISUALISATION APRÈS RESAMPLING — projections PCA
    # ──────────────────────────────────────────────────────────────────────────────
    if plots:
        Z_new_full = pca.transform(scaler_prog_var.transform(X_new).data.detach().cpu().numpy())

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
        fig.suptitle('PC distributions avant / après resampling')
        if n_plots == 1:
            axes = [axes]

        for local_idx in range(n_plots):
            d = components_0indexed[local_idx]
            axes[local_idx].hist(Z_full[:, d], bins=50, alpha=0.6, density=True,
                         color='steelblue', label='avant', edgecolor='none')
            axes[local_idx].hist(Z_new_full[:, d], bins=50, alpha=0.6, density=True,
                         color='seagreen', label='après', edgecolor='none')
            axes[local_idx].set_title(f'PC{d+1}')
            axes[local_idx].set_xlabel(f'PC{d+1}')
            axes[local_idx].set_ylabel('densité')
            axes[local_idx].legend()

        plt.tight_layout()
        plt.show()

    return X_new, Y_new
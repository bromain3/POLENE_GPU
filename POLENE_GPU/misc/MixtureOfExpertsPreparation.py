"""
moeDataPreparation.py
─────────────────────
Utilities for splitting a dataset into k sub-datasets using K-Means
clustering, ready for Mixture-of-Experts training.

Clustering is always done in a dedicated space: a `clustering_scaler` is
fitted on the full dataset and used *only* for K-Means assignment.  It is
returned so the caller can pass it to MixtureOfExperts for routing at
inference time.

Two public functions
--------------------
prepare_moe_datasets_shared_scaler(
        Xdtb, Ydtb, clustering_scaler, expert_scaler_cls, n_clusters)
    • Fits `clustering_scaler` on full Xdtb  →  K-Means split.
    • Fits a single `expert_scaler_cls()` on the full Xdtb.
    • Every expert receives a deep copy of that global scaler (same stats,
      no per-cluster refit).
    • Returns (clusterer, clustering_scaler, X_list, Y_list, scaler_list)
      where scaler_list[k] is a copy of the global expert scaler.

prepare_moe_datasets_per_cluster_scaler(
        Xdtb, Ydtb, clustering_scaler, expert_scaler_cls, n_clusters)
    • Fits `clustering_scaler` on full Xdtb  →  K-Means split.
    • For each cluster, instantiates a fresh `expert_scaler_cls()` and
      fits it on that cluster's X sub-dataset.
    • Returns (clusterer, clustering_scaler, X_list, Y_list, scaler_list)
      where scaler_list[k] is a scaler fitted on cluster k only.

Both functions return raw (unscaled) sub-datasets so that the caller
(e.g. a trainer loop) can apply whatever scaling is appropriate.
"""

import copy
import torch
import numpy as np
from sklearn.cluster import KMeans


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _kmeans_split(X_scaled_np, Xdtb, Ydtb, n_clusters, random_state):
    """
    Run K-Means on a numpy array and split Xdtb / Ydtb by cluster.

    Les clusters sont relabellisés pour que le label 0 soit le plus petit
    cluster et (n_clusters - 1) le plus grand. Le KMeans fitté est mis à
    jour en place (cluster_centers_ et labels_) pour que tout futur appel
    à clusterer.predict(...) renvoie directement des labels cohérents
    avec l'ordre de X_list / Y_list / scaler_list (et donc avec la liste
    `experts` construite à partir de ces listes).

    Returns
    -------
    clusterer : KMeans fitté, relabellisé par taille croissante
    X_list    : list[serie]  — un par cluster, ordonné par taille croissante
    Y_list    : list[serie]  — idem
    labels    : np.ndarray   — label de cluster relabellisé par échantillon
    """
    clusterer  = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    raw_labels = clusterer.fit_predict(X_scaled_np)

    # order[new_label] = old_label, trié par taille croissante
    sizes = np.array([(raw_labels == old).sum() for old in range(n_clusters)])
    order = np.argsort(sizes)

    old_to_new = np.empty(n_clusters, dtype=int)
    old_to_new[order] = np.arange(n_clusters)
    labels = old_to_new[raw_labels]

    # Relabellise l'estimateur lui-même pour que predict() sur de
    # nouvelles données ressorte déjà dans le bon espace de labels
    clusterer.cluster_centers_ = clusterer.cluster_centers_[order]
    clusterer.labels_          = labels

    X_list, Y_list = [], []
    for k in range(n_clusters):
        mask    = torch.from_numpy(labels == k)
        indices = mask.nonzero(as_tuple=True)[0]
        X_list.append(Xdtb[indices])
        Y_list.append(Ydtb[indices])

    return clusterer, X_list, Y_list, labels


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def prepare_moe_datasets_shared_scaler(
    Xdtb,
    Ydtb,
    clustering_scaler,
    expert_scaler_cls,
    n_clusters,
    random_state=42,
):
    """
    Shared-scaler variant: one scaler fitted on the full dataset, copied to
    every expert.

    Parameters
    ----------
    Xdtb               : serie  — full input dataset
    Ydtb               : serie  — full output dataset
    clustering_scaler  : scaler instance — fitted on full Xdtb, used only for
                         K-Means routing (stored in MixtureOfExperts)
    expert_scaler_cls  : class (not instance) — e.g. stdScaler — instantiated
                         once, fitted on full Xdtb, then deep-copied per expert
    n_clusters         : int
    random_state       : int

    Returns
    -------
    clusterer          : fitted KMeans
    clustering_scaler  : the fitted routing scaler (pass to MixtureOfExperts)
    X_list             : list[serie]   — raw sub-datasets, one per cluster
    Y_list             : list[serie]   — raw sub-datasets, one per cluster
    scaler_list        : list[scaler]  — deep copies of the global expert scaler
    """
    # 1. Fit clustering scaler and build numpy array for K-Means
    Xdtb = Xdtb.clone()
    clustering_scaler.fit(Xdtb)
    X_scaled_np = clustering_scaler.transform(Xdtb).data.detach().cpu().numpy().astype(np.float64)

    # 2. K-Means split
    clusterer, X_list, Y_list, _ = _kmeans_split(
        X_scaled_np, Xdtb, Ydtb, n_clusters, random_state
    )

    # 3. Fit a single expert scaler on the full dataset
    global_scaler = expert_scaler_cls()
    global_scaler.fit(Xdtb)

    # 4. Give each expert a deep copy (same stats, independent objects)
    scaler_list = [copy.deepcopy(global_scaler) for _ in range(n_clusters)]

    return clusterer, clustering_scaler, X_list, Y_list, scaler_list


def prepare_moe_datasets_per_cluster_scaler(
    Xdtb,
    Ydtb,
    clustering_scaler,
    expert_scaler_cls,
    n_clusters,
    random_state=42,
):
    """
    Per-cluster-scaler variant: one independent scaler fitted on each cluster.

    Parameters
    ----------
    Xdtb               : serie  — full input dataset
    Ydtb               : serie  — full output dataset
    clustering_scaler  : scaler instance — fitted on full Xdtb, used only for
                         K-Means routing (stored in MixtureOfExperts)
    expert_scaler_cls  : class (not instance) — e.g. stdScaler — instantiated
                         and fitted independently for each cluster's X sub-dataset
    n_clusters         : int
    random_state       : int

    Returns
    -------
    clusterer          : fitted KMeans
    clustering_scaler  : the fitted routing scaler (pass to MixtureOfExperts)
    X_list             : list[serie]   — raw sub-datasets, one per cluster
    Y_list             : list[serie]   — raw sub-datasets, one per cluster
    scaler_list        : list[scaler]  — one scaler fitted on cluster k's X
    """
    # 1. Fit clustering scaler and build numpy array for K-Means
    Xdtb = Xdtb.clone()
    clustering_scaler.fit(Xdtb)
    X_scaled_np = clustering_scaler.transform(Xdtb).data.detach().cpu().numpy().astype(np.float64)

    # 2. K-Means split
    clusterer, X_list, Y_list, _ = _kmeans_split(
        X_scaled_np, Xdtb, Ydtb, n_clusters, random_state
    )

    # 3. Fit a fresh scaler on each cluster's X sub-dataset
    scaler_list = []
    for X_sub in X_list:
        scaler = expert_scaler_cls()
        scaler.fit(X_sub)
        scaler_list.append(scaler)

    return clusterer, clustering_scaler, X_list, Y_list, scaler_list
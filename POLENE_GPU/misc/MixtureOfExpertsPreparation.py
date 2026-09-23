"""
moeDataPreparation.py
─────────────────────
Utilities for splitting a dataset into k sub-datasets using K-Means
clustering, ready for Mixture-of-Experts training.

Clustering is always done in a dedicated space: a `clustering_scaler` is
fitted on the structuring dataset (`X_struct`, which defaults to `X` itself)
and used *only* for K-Means assignment. It is returned so the caller can
pass it to MixtureOfExperts for routing at inference time.

The clustering computation itself is fully factored out into
`_compute_clusters`, so that every public function below builds its split
from the exact same labels — no risk of the partition silently differing
between the "shared scaler" and "per-cluster scaler" variants.

Three public functions
-----------------------
prepare_moe_datasets_shared_scaler(
        X, Y, clustering_scaler, expert_scaler_cls, n_clusters, X_struct=None)
    • Fits `clustering_scaler` on X_struct (or X if not given)  →  K-Means split.
    • Fits a single `expert_scaler_cls()` on the full X.
    • Every expert receives a deep copy of that global scaler (same stats,
      no per-cluster refit).
    • Returns (clusterer, clustering_scaler, X_list, Y_list, scaler_list)
      where scaler_list[k] is a copy of the global expert scaler.

prepare_moe_datasets_per_cluster_scaler(
        X, Y, clustering_scaler, expert_scaler_cls, n_clusters, X_struct=None)
    • Fits `clustering_scaler` on X_struct (or X if not given)  →  K-Means split.
    • For each cluster, instantiates a fresh `expert_scaler_cls()` and
      fits it on that cluster's X sub-dataset.
    • Returns (clusterer, clustering_scaler, X_list, Y_list, scaler_list)
      where scaler_list[k] is a scaler fitted on cluster k only.

assign_clusters(X_struct, clustering_scaler, clusterer)
    • Convenience helper to route new/validation samples to the same
      cluster labels at inference/validation time (kept here so the
      scaling/casting convention stays consistent with fitting).

Both preparation functions return raw (unscaled) sub-datasets so that the
caller (e.g. a trainer loop) can apply whatever scaling is appropriate.

Note on X_struct
----------------
X_struct lets you cluster in a different feature space than the one used
to train the experts (e.g. cluster on a reduced/structural representation,
but train each expert on the full X). X_struct and X must have the same
number of samples and be sample-aligned (X_struct[i] <-> X[i] <-> Y[i]).
If X_struct is None, X is used for clustering (previous behaviour).
"""

import copy
import torch
import numpy as np
from sklearn.cluster import KMeans


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _kmeans_relabel_by_size(raw_labels, n_clusters):
    """
    Relabellise les labels bruts de KMeans pour que le label 0 corresponde
    au plus petit cluster et (n_clusters - 1) au plus grand.

    Returns
    -------
    labels      : np.ndarray — labels relabellisés par taille croissante
    order       : np.ndarray — order[new_label] = old_label
    """
    sizes = np.array([(raw_labels == old).sum() for old in range(n_clusters)])
    order = np.argsort(sizes)

    old_to_new = np.empty(n_clusters, dtype=int)
    old_to_new[order] = np.arange(n_clusters)
    labels = old_to_new[raw_labels]

    return labels, order


def _compute_clusters(X_struct, clustering_scaler, n_clusters, random_state):
    """
    Fit `clustering_scaler` on X_struct and run K-Means on the result.

    C'est le SEUL endroit où le clustering est calculé. Toutes les
    fonctions publiques passent par ici, ce qui garantit qu'elles
    produisent exactement la même partition pour les mêmes entrées.

    Les clusters sont relabellisés pour que le label 0 soit le plus petit
    cluster et (n_clusters - 1) le plus grand. Le KMeans fitté est mis à
    jour en place (cluster_centers_ et labels_) pour que tout futur appel
    à clusterer.predict(...) renvoie directement des labels cohérents
    avec cet ordre.

    Returns
    -------
    clusterer          : KMeans fitté, relabellisé par taille croissante
    clustering_scaler  : le scaler fitté (même objet que celui passé en entrée)
    labels             : np.ndarray — label de cluster relabellisé, un par
                         échantillon de X_struct
    """
    X_struct = X_struct.clone()
    clustering_scaler.fit(X_struct)
    X_scaled_np = (
        clustering_scaler.transform(X_struct)
        .data.detach().cpu().numpy().astype(np.float64)
    )

    clusterer  = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    raw_labels = clusterer.fit_predict(X_scaled_np)

    labels, order = _kmeans_relabel_by_size(raw_labels, n_clusters)

    # Relabellise l'estimateur lui-même pour que predict() sur de
    # nouvelles données ressorte déjà dans le bon espace de labels
    clusterer.cluster_centers_ = clusterer.cluster_centers_[order]
    clusterer.labels_          = labels

    return clusterer, clustering_scaler, labels


def _split_by_labels(X, Y, labels, n_clusters):
    """
    Découpe X et chaque série de Y en sous-series selon `labels`.

    X et chaque élément de Y doivent être alignés échantillon par
    échantillon avec `labels` (même nombre de samples, même ordre).

    Parameters
    ----------
    X      : serie
    Y      : list[serie] — plusieurs séries alignées sur les mêmes échantillons
    labels : np.ndarray
    n_clusters : int

    Returns
    -------
    X_list : list[serie]        — un sous-dataset X par cluster
    Y_list : list[list[serie]]  — Y_list[k] est la liste des sous-series de
                                  Y (une par élément de la liste d'origine)
                                  restreintes aux échantillons du cluster k
    """
    X_list, Y_list = [], []
    for k in range(n_clusters):
        mask    = torch.from_numpy(labels == k)
        indices = mask.nonzero(as_tuple=True)[0]
        X_list.append(X[indices])
        Y_list.append([y[indices] for y in Y])

    return X_list, Y_list


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def assign_clusters(X_struct, clustering_scaler, clusterer):
    """
    Route de nouveaux échantillons (typiquement un jeu de validation) vers
    les labels de cluster déjà appris, en utilisant le même scaler que lors
    du fit.

    Parameters
    ----------
    X_struct           : serie — données dans le même espace que celui
                         utilisé pour fitter `clustering_scaler` / `clusterer`
    clustering_scaler  : scaler déjà fitté (retourné par une des fonctions
                         prepare_moe_datasets_*)
    clusterer          : KMeans déjà fitté (retourné par une des fonctions
                         prepare_moe_datasets_*)

    Returns
    -------
    labels : np.ndarray — un label de cluster par échantillon de X_struct
    """
    X_scaled_np = (
        clustering_scaler.transform(X_struct)
        .data.detach().cpu().numpy().astype(np.float64)
    )
    return clusterer.predict(X_scaled_np)


def prepare_moe_datasets_shared_scaler(
    X,
    Y,
    clustering_scaler,
    expert_scaler_cls,
    n_clusters,
    X_struct=None,
    random_state=42,
):
    """
    Shared-scaler variant: one scaler fitted on the full dataset, copied to
    every expert.

    Parameters
    ----------
    X                   : serie  — full input dataset (used to train experts)
    Y                   : serie  — full output dataset
    clustering_scaler   : scaler instance — fitted on X_struct (or X), used
                          only for K-Means routing (stored in MixtureOfExperts)
    expert_scaler_cls   : class (not instance) — e.g. stdScaler — instantiated
                          once, fitted on full X, then deep-copied per expert
    n_clusters          : int
    X_struct            : serie or None — dataset used to fit the clustering
                          space. Must be sample-aligned with X and Y (same
                          nSamples, same order). If None, X is used (i.e.
                          clustering and expert features are the same space).
    random_state        : int

    Returns
    -------
    clusterer          : fitted KMeans
    clustering_scaler  : the fitted routing scaler (pass to MixtureOfExperts)
    X_list             : list[serie]   — raw sub-datasets, one per cluster
    Y_list             : list[list[serie]]  — Y_list[k] is the list of
                          sub-series (one per element of the input Y list)
                          restricted to cluster k's samples
    scaler_list        : list[scaler]  — deep copies of the global expert scaler
    """
    if X_struct is None:
        X_struct = X

    clusterer, clustering_scaler, labels = _compute_clusters(
        X_struct, clustering_scaler, n_clusters, random_state
    )
    X_list, Y_list = _split_by_labels(X, Y, labels, n_clusters)

    # Fit a single expert scaler on the full dataset, then deep-copy it
    # for each expert (same stats, independent objects)
    global_scaler = expert_scaler_cls()
    global_scaler.fit(X)
    scaler_list = [copy.deepcopy(global_scaler) for _ in range(n_clusters)]

    return clusterer, clustering_scaler, X_list, Y_list, scaler_list


def prepare_moe_datasets_per_cluster_scaler(
    X,
    Y,
    clustering_scaler,
    expert_scaler_cls,
    n_clusters,
    X_struct=None,
    random_state=42,
):
    """
    Per-cluster-scaler variant: one independent scaler fitted on each cluster.

    Parameters
    ----------
    X                   : serie  — full input dataset (used to train experts)
    Y                   : serie  — full output dataset
    clustering_scaler   : scaler instance — fitted on X_struct (or X), used
                          only for K-Means routing (stored in MixtureOfExperts)
    expert_scaler_cls   : class (not instance) — e.g. stdScaler — instantiated
                          and fitted independently for each cluster's X
                          sub-dataset
    n_clusters          : int
    X_struct            : serie or None — dataset used to fit the clustering
                          space. Must be sample-aligned with X and Y (same
                          nSamples, same order). If None, X is used (i.e.
                          clustering and expert features are the same space).
    random_state        : int

    Returns
    -------
    clusterer          : fitted KMeans
    clustering_scaler  : the fitted routing scaler (pass to MixtureOfExperts)
    X_list             : list[serie]   — raw sub-datasets, one per cluster
    Y_list             : list[list[serie]]  — Y_list[k] is the list of
                          sub-series (one per element of the input Y list)
                          restricted to cluster k's samples
    scaler_list        : list[scaler]  — one scaler fitted on cluster k's X
    """
    if X_struct is None:
        X_struct = X

    clusterer, clustering_scaler, labels = _compute_clusters(
        X_struct, clustering_scaler, n_clusters, random_state
    )
    X_list, Y_list = _split_by_labels(X, Y, labels, n_clusters)

    # Fit a fresh scaler on each cluster's X sub-dataset
    scaler_list = []
    for X_sub in X_list:
        scaler = expert_scaler_cls()
        scaler.fit(X_sub)
        scaler_list.append(scaler)

    return clusterer, clustering_scaler, X_list, Y_list, scaler_list
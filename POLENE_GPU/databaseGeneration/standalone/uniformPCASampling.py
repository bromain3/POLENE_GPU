"""
pca_uniform_sampler.py
======================
Fit a PCA on a chemical state manifold, then draw N samples
uniformly (or quasi-uniformly) inside the latent bounding box
[PC1_min, PC1_max] × … × [PCk_min, PCk_max] and back-project
them to the scaled and physical spaces.

Physical constraints (fractions massiques)
------------------------------------------
X[:,0] = T  (temperature)  →  x >= 0  only
X[:,1] = P  (pressure)     →  x >= 0  only
X[:,2:]     = mass fractions → x_i >= 0  AND  Σ x_i = 1

Three enforcement strategies are available via the ``constraint`` parameter
of ``.sample()``:

    'none'    – no enforcement (original behaviour)
    'clamp'   – clip negatives to 0, then L1-normalise the mass fractions only
    'project' – orthogonal projection of the mass-fraction block onto the
                probability simplex via the Duchi et al. (2008) sort-based
                algorithm (exact, unbiased); T and P are merely clipped to 0
    'reject'  – rejection sampling: keep only rows that already satisfy the
                constraints within tolerances, and redraw the rest
                (unbiased but may be slow when the feasible fraction is low)

Public API
----------
PCAUniformSampler
    .fit(X)
    .sample(n_samples, mode='uniform', constraint='project') → serie (N_new, F)
    .fit_sample(X, n_samples, ...)       convenience wrapper

Standalone helpers (mirror the original plot* functions)
    plotLoadings_sampler(...)
    plot2DSlice_sampler(...)
    plotPChistogram_sampler(...)
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from POLENE_GPU.dataHandler.serie import serie

# Column layout convention
_N_EXTRA = 2   # number of leading non-fraction columns (T, P)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _to_numpy(X) -> np.ndarray:
    """Accept ndarray, PyTorch tensor, or POLENE object with .data.detach()."""
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


def _pc_label(d: int, evr: np.ndarray | None) -> str:
    base = f"PC{d + 1}"
    if evr is not None and d < len(evr):
        return f"{base}  ({evr[d] * 100:.1f}%)"
    return base


# ──────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTRAINT ENFORCEMENT
# ──────────────────────────────────────────────────────────────────────────────

def _project_simplex_rows(v: np.ndarray) -> np.ndarray:
    """
    Orthogonal projection of each row of *v* onto the probability simplex
        Δ = { x ∈ ℝ^F : x_i ≥ 0,  Σ x_i = 1 }

    Algorithm: sort-based O(F log F), Duchi et al. (2008).
    "Efficient Projections onto the L1-Ball for Learning in High Dimensions."
    ICML 2008.

    Parameters
    ----------
    v : (N, F) array  — mass-fraction block only

    Returns
    -------
    w : (N, F) array projected onto Δ
    """
    N, F = v.shape
    u = np.sort(v, axis=1)[:, ::-1]                       # (N, F) descending
    cssv = np.cumsum(u, axis=1)                            # (N, F)
    rho = (u > (cssv - 1.0) / np.arange(1, F + 1)).sum(axis=1) - 1  # (N,)
    theta = (cssv[np.arange(N), rho] - 1.0) / (rho + 1.0)            # (N,)
    return np.maximum(v - theta[:, np.newaxis], 0.0)


def _project_simplex(X: np.ndarray) -> np.ndarray:
    """
    Enforce physical constraints on a full sample matrix X (N, F):
      - columns 0..(_N_EXTRA-1)  : T, P  → clipped to 0 (must be positive)
      - columns _N_EXTRA..        : mass fractions → projected onto simplex
                                    (x_i >= 0, Σ = 1)
    """
    X = X.copy()
    # T and P: positivity only
    X[:, :_N_EXTRA] = np.maximum(X[:, :_N_EXTRA], 0.0)
    # Mass fractions: simplex projection
    X[:, _N_EXTRA:] = _project_simplex_rows(X[:, _N_EXTRA:])
    return X


def _clamp_normalize(X: np.ndarray) -> np.ndarray:
    """
    Fast but slightly biased enforcement:
      - columns 0..(_N_EXTRA-1)  : T, P → clipped to 0
      - columns _N_EXTRA..        : mass fractions → clipped to 0,
                                    then L1-renormalised to sum = 1

    If an entire mass-fraction row is zero after clipping (very rare),
    a warning is raised and the row is replaced with a uniform composition.
    """
    X = X.copy()
    # T and P: positivity only
    X[:, :_N_EXTRA] = np.maximum(X[:, :_N_EXTRA], 0.0)

    # Mass fractions
    fracs = np.maximum(X[:, _N_EXTRA:], 0.0)
    row_sums = fracs.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).ravel()
    if zero_rows.any():
        warnings.warn(
            f"{zero_rows.sum()} back-projected points had all-zero mass-fraction "
            "components after clipping; replaced with uniform composition.",
            RuntimeWarning,
            stacklevel=3,
        )
        n_fracs = fracs.shape[1]
        fracs[zero_rows] = 1.0 / n_fracs
        row_sums[zero_rows] = 1.0
    X[:, _N_EXTRA:] = fracs / row_sums
    return X


def _is_valid(X: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Boolean mask: True iff row i satisfies:
      - x_i >= -tol                 for ALL features (T, P, and mass fracs)
      - |Σ_{i>=_N_EXTRA} x_i - 1| <= tol   (mass fractions sum to 1)
    """
    non_neg = (X >= -tol).all(axis=1)
    unit_sum = np.abs(X[:, _N_EXTRA:].sum(axis=1) - 1.0) <= tol
    return non_neg & unit_sum


# ──────────────────────────────────────────────────────────────────────────────
# SAMPLING STRATEGIES (latent space)
# ──────────────────────────────────────────────────────────────────────────────

def _sample_uniform_box(n: int, mins: np.ndarray, maxs: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Pure uniform draw inside the hyper-rectangle."""
    k = len(mins)
    return mins + (maxs - mins) * rng.random((n, k))


def _sample_lhs(n: int, mins: np.ndarray, maxs: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling — one sample per stratum on each axis."""
    k = len(mins)
    samples = np.empty((n, k))
    for j in range(k):
        perm = rng.permutation(n)
        u = (perm + rng.random(n)) / n
        samples[:, j] = mins[j] + (maxs[j] - mins[j]) * u
    return samples


def _sample_sobol(n: int, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Quasi-random low-discrepancy Sobol sequence."""
    try:
        from scipy.stats.qmc import Sobol
        engine = Sobol(d=len(mins), scramble=True)
        u = engine.random(n)
    except ImportError:
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                  31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        k = len(mins)

        def vdc(n_seq: int, base: int) -> np.ndarray:
            seq = np.zeros(n_seq)
            for i in range(1, n_seq + 1):
                f, r, ni = 1 / base, 0.0, i
                while ni > 0:
                    f /= base
                    r += f * (ni % base)
                    ni //= base
                seq[i - 1] = r
            return seq

        u = np.column_stack([vdc(n, primes[j % len(primes)]) for j in range(k)])

    return mins + (maxs - mins) * u


def _draw_latent(n: int, mins: np.ndarray, maxs: np.ndarray,# TODO: ajouter une option pour soit sampler uniformément soit suivant une gaussienne car il semble que les PC ont tendance à gaussianiser les distributions (à s'approcher d'un cercle en outre) et c'est ce qu'on observe
                 mode: str, rng: np.random.Generator) -> np.ndarray:
    """Dispatch to the right latent-space sampler."""
    if mode == "lhs":
        return _sample_lhs(n, mins, maxs, rng)
    elif mode == "sobol":
        return _sample_sobol(n, mins, maxs)
    else:
        return _sample_uniform_box(n, mins, maxs, rng)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ──────────────────────────────────────────────────────────────────────────────

class PCAUniformSampler:
    """
    Fit a PCA on a chemical state manifold and draw uniformly distributed
    samples across the principal-component bounding box, then back-project
    to the physical space while enforcing the physical constraints:

    Column layout of X
    ------------------
    X[:, 0]       : T (temperature)   → x >= 0
    X[:, 1]       : P (pressure)      → x >= 0
    X[:, 2:]      : mass fractions    → x_i >= 0  AND  Σ x_i = 1

    Parameters
    ----------
    n_components : int
        Number of principal components (K).
    scaler : sklearn-compatible scaler
        Must implement fit_transform / inverse_transform.
        Defaults to StandardScaler().
    random_state : int | None
        Seed for reproducibility.

    Attributes (after .fit)
    -----------------------
    pca_          : fitted sklearn PCA object
    scaler_       : fitted scaler
    pc_mins_      : (K,) lower bound of each PC over training data
    pc_maxs_      : (K,) upper bound of each PC over training data
    explained_variance_ratio_ : (K,) EVR per component
    """

    def __init__(
        self,
        n_components: int,
        scaler=None,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.random_state = random_state

        self.pca_: PCA | None = None
        self.scaler_ = None
        self.pc_mins_: np.ndarray | None = None
        self.pc_maxs_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self._rng = np.random.default_rng(random_state)

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, prints: bool = True) -> "PCAUniformSampler":
        """
        Scale X then fit PCA.  Stores PC bounding box from training data.

        Parameters
        ----------
        X      : array-like (N, F)  — columns: [T, P, w_1, …, w_M]
        prints : print summary to stdout
        """
        X_np = _to_numpy(X)

        # 1. scale
        self.scaler_ = self.scaler
        X_scaled = self.scaler_.fit_transform(X)
        self.colNames = X.colName
        self.device = X.storingDevice
        self.dtype = X.storingDtype

        # 2. PCA
        self.pca_ = PCA(n_components=self.n_components)
        Z = self.pca_.fit_transform(X_scaled.data.detach().cpu().numpy())   # (N, K)

        # 3. latent bounding box
        self.pc_mins_ = Z.min(axis=0)
        self.pc_maxs_ = Z.max(axis=0)
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_

        if prints:
            evr_cum = self.explained_variance_ratio_.cumsum()
            print(f"PCAUniformSampler fitted  |  N={X_np.shape[0]}  "
                  f"F={X_np.shape[1]}  K={self.n_components}")
            print(f"Cumulative EVR: {evr_cum[-1] * 100:.1f}%")
            for k, (ev, lo, hi) in enumerate(
                zip(self.explained_variance_ratio_, self.pc_mins_, self.pc_maxs_)
            ):
                print(f"  PC{k+1:02d}  EVR={ev*100:.2f}%  range=[{lo:.4f}, {hi:.4f}]")

        return self

    # ── internal back-projection ──────────────────────────────────────────────

    def _backproject(self, Z: np.ndarray) -> np.ndarray:
        """
        Latent Z (M, K) → physical numpy array (M, F).
        Does NOT enforce any constraint; that is the caller's responsibility.
        """
        X_scaled_rec = self.pca_.inverse_transform(Z)                      # (M, F)
        X_scaled_serie = serie(
            X_scaled_rec, self.colNames,
            storingDevice=self.device, storingDtype=self.dtype,
        )
        X_phys = self.scaler_.inverse_transform(X_scaled_serie)
        return _to_numpy(X_phys)

    # ── sample ────────────────────────────────────────────────────────────────

    def sample(
        self,
        n_samples: int,
        mode: str = "uniform",
        constraint: str = "project",
        reject_max_iter: int = 50,
        reject_tol: float = 1e-6,
        prints: bool = True,
    ):
        """
        Draw ``n_samples`` in the latent box, back-project, and enforce
        physical constraints.

        Column layout enforced
        ----------------------
        X[:, 0]   T  (temperature) → x >= 0
        X[:, 1]   P  (pressure)    → x >= 0
        X[:, 2:]  mass fractions   → x_i >= 0  AND  Σ x_i = 1

        Parameters
        ----------
        n_samples      : int
        mode           : {'uniform', 'lhs', 'sobol'}
        constraint     : {'none', 'clamp', 'project', 'reject'}
            'none'    – no enforcement.
            'clamp'   – clip all values to 0; L1-renormalise mass fracs only.
            'project' – simplex projection on mass-frac block; clip T and P.
                        *Default.*
            'reject'  – keep only rows that naturally satisfy constraints;
                        redraw the rest.
        reject_max_iter : int
        reject_tol     : float
        prints         : bool

        Returns
        -------
        X_phys : serie  shape (n_samples, F)
        """
        if self.pca_ is None:
            raise RuntimeError("Call .fit(X) before .sample().")

        constraint = constraint.lower()
        if constraint not in {"none", "clamp", "project", "reject"}:
            raise ValueError(f"Unknown constraint={constraint!r}. "
                             "Choose from 'none', 'clamp', 'project', 'reject'.")

        # ── branch: deterministic post-processing strategies ─────────────────
        if constraint in {"none", "clamp", "project"}:
            Z_new = _draw_latent(n_samples, self.pc_mins_, self.pc_maxs_,
                                 mode, self._rng)
            X_phys = self._backproject(Z_new)

            if constraint == "clamp":
                X_phys = _clamp_normalize(X_phys)
            elif constraint == "project":
                X_phys = _project_simplex(X_phys)
            # else 'none': leave as-is

        # ── branch: rejection sampling ────────────────────────────────────────
        else:  # constraint == 'reject'
            collected = []
            n_needed = n_samples
            total_drawn = 0
            n_accepted = 0

            for iteration in range(reject_max_iter):
                if n_needed == 0:
                    break

                n_draw = max(n_needed * 4, 256)
                Z_try = _draw_latent(n_draw, self.pc_mins_, self.pc_maxs_,
                                     mode, self._rng)
                X_try = self._backproject(Z_try)
                total_drawn += n_draw

                valid = _is_valid(X_try, tol=reject_tol)
                n_new = valid.sum()

                if n_new > 0:
                    take = min(n_new, n_needed)
                    collected.append(X_try[valid][:take])
                    n_accepted += take
                    n_needed -= take

            if n_needed > 0:
                feasible_rate = n_accepted / max(total_drawn, 1)
                warnings.warn(
                    f"Rejection sampling converged with only {n_accepted}/{n_samples} "
                    f"valid samples after {reject_max_iter} iterations "
                    f"(feasible rate ≈ {feasible_rate:.2%}). "
                    "Consider switching to constraint='project' or 'clamp', "
                    "or increasing reject_max_iter.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                n_missing = n_needed
                Z_pad = _draw_latent(n_missing * 2, self.pc_mins_, self.pc_maxs_,
                                     mode, self._rng)
                X_pad = _project_simplex(self._backproject(Z_pad))[:n_missing]
                collected.append(X_pad)

            X_phys = np.vstack(collected)[:n_samples]

        if prints:
            neg_frac = (X_phys < 0).any(axis=1).mean() * 100
            frac_sum_dev = np.abs(X_phys[:, _N_EXTRA:].sum(axis=1) - 1.0).max()
            print(
                f"Sampled {n_samples} pts  |  mode={mode}  constraint={constraint}  "
                f"output shape={X_phys.shape}\n"
                f"  Rows with any negative component : {neg_frac:.2f}%  "
                f"(should be 0 unless constraint='none')\n"
                f"  Max |Σ mass_fracs - 1| : {frac_sum_dev:.2e}  "
                f"(should be ~0 unless constraint='none')"
            )

        return serie(X_phys, self.colNames,
                     storingDevice=self.device, storingDtype=self.dtype)

    # ── convenience wrapper ───────────────────────────────────────────────────

    def fit_sample(
        self,
        X,
        n_samples: int,
        mode: str = "uniform",
        constraint: str = "project",
        prints: bool = True,
    ):
        """Fit then sample in one call.  Returns a serie of shape (N, F)."""
        return self.fit(X, prints=prints).sample(
            n_samples, mode=mode, constraint=constraint, prints=prints
        )

    # ── projection helpers (public) ───────────────────────────────────────────

    def transform(self, X) -> np.ndarray:
        """Project X into the latent PC space. Returns Z (N, K)."""
        if self.pca_ is None:
            raise RuntimeError("Call .fit(X) first.")
        X_scaled = self.scaler_.transform(_to_numpy(X))
        return self.pca_.transform(X_scaled)

    def inverse_transform_latent(self, Z: np.ndarray) -> np.ndarray:
        """Back-project latent coords Z (M, K) → physical space (M, F)."""
        if self.pca_ is None:
            raise RuntimeError("Call .fit(X) first.")
        return self.scaler_.inverse_transform(self.pca_.inverse_transform(Z))
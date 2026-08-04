from POLENE_GPU.preProcessing.scaler._BaseTransform import _BaseTransform
from POLENE_GPU.dataHandler.serie import serie

import numpy as np
import torch


class softlogScaler(_BaseTransform):
    """
    Transformation softlog, bijective et C1 :

        f(x) = ln(x / tau)     si x < tau
               x / tau - 1     si x >= tau

    tau peut être :
        - un scalaire (même tau pour toutes les espèces)
        - un vecteur (un tau par espèce)
        - None -> recherche automatique au fit() (même logique que l'ancien
          SoftLogTransform.automaticTauFindingAll : balayage log-spacé des
          candidats + raffinement local, scoré par entropie ou flatness de
          l'histogramme après standardisation).
    """

    def __init__(self, threshold, tau=None,
                 nbins=150, n_tau_candidates=50,
                 tau_percentile_range=(0, 100), metric='entropy',
                 verbose=False):
        self.threshold = threshold
        self.tau = tau  # None, float, ou array-like (n_species,)

        # hyperparamètres de la recherche auto de tau
        self.nbins = nbins
        self.n_tau_candidates = n_tau_candidates
        self.tau_percentile_range = tau_percentile_range
        self.metric = metric
        self.verbose = verbose

        self._species_mask = None
        self._col_names = None
        self._device = None
        self._dtype = None
        self._tau_tensor = None  # (n_species,), aligné à l'ordre des colonnes masquées

    # ------------------------------------------------------------------
    # Helpers communs (identiques à logScaler)
    # ------------------------------------------------------------------

    def _build_species_mask(self, col_names, device):
        return torch.tensor(
            [name not in self.NON_SPECIES_COLS for name in col_names],
            dtype=torch.bool,
            device=device,
        )

    def _sync(self, x):
        self._species_mask = self._species_mask.to(x.storingDevice)
        self._tau_tensor = self._tau_tensor.to(x.storingDevice).to(x.storingDtype)
        self._device = x.storingDevice
        self._dtype = x.storingDtype

    def _syncTorch(self, x):
        self._species_mask = self._species_mask.to(x.device)
        self._tau_tensor = self._tau_tensor.to(x.device).to(x.dtype)
        self._device = x.device
        self._dtype = x.dtype

    def _threshold_tensor(self, device, dtype):
        return torch.tensor(self.threshold, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Recherche automatique de tau (portage de l'ancien code)
    # ------------------------------------------------------------------

    def _automatic_tau_finding_1d(self, y: np.ndarray) -> float:
        from sklearn.preprocessing import StandardScaler as _StandardScaler

        y = np.asarray(y, dtype=float).ravel()
        scaler = _StandardScaler()

        p_low, p_high = self.tau_percentile_range
        y_positive = y[y > self.threshold]
        if len(y_positive) == 0:
            return self.threshold * 10.0

        tau_low = np.percentile(y_positive, p_low)
        tau_high = np.percentile(y_positive, p_high)

        if tau_low <= 0 or tau_high <= 0 or tau_low >= tau_high:
            return float(np.median(y_positive))

        tau_candidates = np.logspace(np.log10(tau_low), np.log10(tau_high), self.n_tau_candidates)

        def _score(tau_val):
            x_clip = np.where(y <= self.threshold, self.threshold, y)
            below = x_clip < tau_val
            y_t = np.where(below, np.log(x_clip / tau_val), x_clip / tau_val - 1.0)
            y_s = scaler.fit_transform(y_t.reshape(-1, 1)).ravel()
            y_c = y_s[~np.isnan(y_s) & ~np.isinf(y_s)]
            counts, _ = np.histogram(y_c, bins=self.nbins)
            counts_nz = counts[counts > 0]
            if len(counts_nz) == 0:
                return np.inf
            if self.metric == 'flatness':
                return counts_nz.max() / counts_nz.mean()
            probs = counts_nz / counts_nz.sum()
            return -(-np.sum(probs * np.log(probs)) / np.log(self.nbins))

        scores = np.array([_score(t) for t in tau_candidates])
        best_idx = int(np.argmin(scores))

        idx_lo = max(0, best_idx - 1)
        idx_hi = min(len(tau_candidates) - 1, best_idx + 1)
        tau_refine = np.logspace(
            np.log10(tau_candidates[idx_lo]),
            np.log10(tau_candidates[idx_hi]),
            self.n_tau_candidates,
        )
        scores_refine = np.array([_score(t) for t in tau_refine])
        best_idx_r = int(np.argmin(scores_refine))
        best_tau = tau_refine[best_idx_r]

        if self.verbose:
            print(f"[softlogScaler] tau optimal trouvé : {best_tau:.6e} "
                  f"({self.metric} = {scores_refine[best_idx_r]:.3f})")

        return float(best_tau)

    def _fit_tau(self, data, col_names):
        """
        Détermine self._tau_tensor (n_species,), aligné à l'ordre des colonnes
        d'espèces (parmi col_names, filtrées par le masque).
        """
        species_names = [n for n in col_names if n not in self.NON_SPECIES_COLS]
        n_species = len(species_names)

        if self.tau is not None:
            tau_arr = np.asarray(self.tau, dtype=float)
            if tau_arr.ndim == 0:
                tau_arr = np.full(n_species, float(tau_arr))
            elif tau_arr.shape[0] != n_species:
                raise ValueError(
                    f"tau fourni de longueur {tau_arr.shape[0]} != nombre d'espèces {n_species}"
                )
            self._tau_tensor = torch.tensor(tau_arr, dtype=data.dtype, device=data.device)
            return

        # recherche automatique, colonne par colonne (même logique que l'ancien code)
        data_np = data[:, self._species_mask].detach().cpu().numpy()
        taus = np.empty(n_species, dtype=float)
        for j in range(n_species):
            if self.verbose:
                print(f"[softlogScaler] recherche de tau pour '{species_names[j]}' "
                      f"({j + 1}/{n_species})")
            taus[j] = self._automatic_tau_finding_1d(data_np[:, j])

        self.tau = taus  # mémorisé (réutilisable / sérialisable)
        self._tau_tensor = torch.tensor(taus, dtype=data.dtype, device=data.device)

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _clip(self, data):
        t = self._threshold_tensor(data.device, data.dtype)
        out = data.clone()
        out[:, self._species_mask] = torch.where(
            data[:, self._species_mask] <= t,
            t,
            data[:, self._species_mask],
        )
        return out

    def _forward(self, data):
        out = data.clone()
        t = self._threshold_tensor(data.device, data.dtype)

        col = data[:, self._species_mask]
        safe_col = torch.where(col <= t, t, col)
        tau = self._tau_tensor

        below = safe_col < tau
        log_branch = torch.log(safe_col / tau)
        linear_branch = safe_col / tau - 1.0
        out[:, self._species_mask] = torch.where(below, log_branch, linear_branch)
        return out

    def _backward(self, data):
        out = data.clone()
        col = data[:, self._species_mask]
        tau = self._tau_tensor

        below = col < 0.0
        exp_branch = tau * torch.exp(col)
        linear_branch = tau * (col + 1.0)
        out[:, self._species_mask] = torch.where(below, exp_branch, linear_branch)
        return out

    # ------------------------------------------------------------------
    # API _BaseTransform
    # ------------------------------------------------------------------

    def fit(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")

        col_names = list(x.colName)
        self._col_names = col_names
        self._species_mask = self._build_species_mask(col_names, x.storingDevice)
        self._device = x.storingDevice
        self._dtype = x.storingDtype

        self._fit_tau(x.data, col_names)
        return self

    def transform(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        return serie(
            self._forward(x.data),
            x.colName,
            storingDevice=x.storingDevice,
            storingDtype=x.storingDtype,
        )

    def transformTorch(self, x):
        self._syncTorch(x)
        return self._forward(x)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        return serie(
            self._backward(x.data),
            x.colName,
            storingDevice=x.storingDevice,
            storingDtype=x.storingDtype,
        )

    def inverse_transformTorch(self, x):
        self._syncTorch(x)
        return self._backward(x)

    # ------------------------------------------------------------------
    # Jacobiennes
    # ------------------------------------------------------------------

    def J(self, x):
        """dy/dx, évaluée en x (espace d'ORIGINE, avant transformation)."""
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        N, F = x.data.shape
        Jm = torch.ones(N, F, device=x.storingDevice, dtype=x.storingDtype)

        t = self._threshold_tensor(x.storingDevice, x.storingDtype)
        col = x.data[:, self._species_mask]
        safe_col = torch.where(col <= t, t, col)
        tau = self._tau_tensor

        below = safe_col < tau
        Jm[:, self._species_mask] = torch.where(below, 1.0 / safe_col, 1.0 / tau)

        return Jm.to(x.storingDevice).to(x.storingDtype)

    def J_1(self, x):
        """dx/dy, évaluée en y (espace TRANSFORMÉ, après transformation)."""
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        N, F = x.data.shape
        J_inv = torch.ones(N, F, device=x.storingDevice, dtype=x.storingDtype)

        col = x.data[:, self._species_mask]
        tau = self._tau_tensor
        below = col < 0.0
        J_inv[:, self._species_mask] = torch.where(below, tau * torch.exp(col), tau)

        return J_inv.to(x.storingDevice).to(x.storingDtype)

    def __repr__(self) -> str:
        return f"softlogScaler(threshold={self.threshold}, tau={self.tau})"
from POLENE_GPU.preProcessing.scaler._BaseTransform import _BaseTransform
from POLENE_GPU.dataHandler.serie import serie

import torch


class asinhScaler(_BaseTransform):
    """
    Transformation y = asinh(a * x / x_q) appliquée par colonne d'espèces.

    x_q est une échelle propre à chaque colonne, calculée au fit :
      - si q est un float dans (0, 1] : x_q = quantile q de |x| de la colonne
      - si q == "max"                 : x_q = max(|x|) de la colonne

    asinh est définie sur tout R (x <= 0 inclus), donc aucun threshold ni clamp
    n'est nécessaire dans le sens direct. Elle est linéaire autour de 0
    (y ~ a*x/x_q pour |a*x/x_q| << 1) et logarithmique pour |a*x/x_q| >> 1.
    La transition se fait autour de x ~ x_q / a.

    Cutoff (sens inverse uniquement) :
      Si cutoff_factor = f est fourni, la transformation inverse clampe y dans
      [-y_max, y_max] avant le sinh, avec
          y_max = asinh(|a| * f * x_max / x_q),   x_max = max|x| de la colonne au fit.
      Cela revient à borner |x| à f * x_max, sans risque d'overflow dans sinh
      (sinh(y) -> inf en float32 dès |y| ~ 89).

      ste=False : clamp dur (gradient nul hors de l'intervalle).
      ste=True  : straight-through estimator. La valeur est clampée en forward,
                  mais le gradient passe comme si le clamp n'existait pas.
                  Utile si la loss est calculée après inverse_transformTorch.
    """

    def __init__(self, q=0.75, a=1.0, cutoff_factor=None, ste=False):
        if a == 0:
            raise ValueError(
                "asinhScaler does not support a=0 (transformation not invertible)."
            )
        if cutoff_factor is not None and cutoff_factor <= 0:
            raise ValueError("cutoff_factor must be > 0 or None.")
        if isinstance(q, str):
            if q.lower() != "max":
                raise ValueError("q must be a float in (0, 1] or the string 'max'.")
            q = "max"
        elif not (0.0 < float(q) <= 1.0):
            raise ValueError("q must be a float in (0, 1] or the string 'max'.")

        self.q = q
        self.a = a
        self.cutoff_factor = cutoff_factor
        self.ste = ste

        self._species_mask = None
        self._col_names = None
        self._scale = None  # x_q par colonne d'espèce, shape (n_species,)
        self._y_max = None  # seuil dans l'espace transformé, shape (n_species,) ou None
        self._device = None
        self._dtype = None

    def _build_species_mask(self, col_names, device):
        return torch.tensor(
            [name not in self.NON_SPECIES_COLS for name in col_names],
            dtype=torch.bool,
            device=device,
        )

    @staticmethod
    def _column_quantile(abs_data, q):
        """Quantile (interpolation linéaire) par colonne, via tri.
        Évite la limite de taille de torch.quantile sur les gros jeux de données."""
        N = abs_data.shape[0]
        sorted_data, _ = torch.sort(abs_data, dim=0)
        pos = q * (N - 1)
        lo = int(torch.floor(torch.tensor(pos)).item())
        hi = int(torch.ceil(torch.tensor(pos)).item())
        frac = pos - lo
        return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac

    def _compute_scale(self, data):
        col = data[:, self._species_mask].abs()

        if self.q == "max":
            scale = col.max(dim=0).values
        else:
            scale = self._column_quantile(col, float(self.q))

        # Garde-fou : colonne dont le quantile vaut 0 (ex. espèce très parcimonieuse)
        # -> on retombe sur le max, puis sur 1.0 si la colonne est entièrement nulle.
        col_max = col.max(dim=0).values
        scale = torch.where(scale > 0, scale, col_max)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return scale

    def _compute_y_max(self, data):
        """Seuil du clamp dans l'espace transformé : y_max = asinh(|a| f x_max / x_q)."""
        if self.cutoff_factor is None:
            return None
        x_max = data[:, self._species_mask].abs().max(dim=0).values
        return torch.asinh(abs(self.a) * self.cutoff_factor * x_max / self._scale)

    def _clamp_y(self, y):
        """Clamp de y dans [-y_max, y_max] (colonnes d'espèces uniquement).
        Avec ste=True : valeur clampée en forward, gradient identité en backward."""
        if self._y_max is None:
            return y
        y_c = torch.maximum(torch.minimum(y, self._y_max), -self._y_max)
        if self.ste:
            return y + (y_c - y).detach()
        return y_c

    def _sync(self, x):
        self._species_mask = self._species_mask.to(x.storingDevice)
        self._scale = self._scale.to(device=x.storingDevice, dtype=x.storingDtype)
        if self._y_max is not None:
            self._y_max = self._y_max.to(device=x.storingDevice, dtype=x.storingDtype)
        self._device = x.storingDevice
        self._dtype = x.storingDtype

    def _syncTorch(self, x):
        self._species_mask = self._species_mask.to(x.device)
        self._scale = self._scale.to(device=x.device, dtype=x.dtype)
        if self._y_max is not None:
            self._y_max = self._y_max.to(device=x.device, dtype=x.dtype)
        self._device = x.device
        self._dtype = x.dtype

    def _forward(self, data):
        out = data.clone()
        out[:, self._species_mask] = torch.asinh(
            self.a * data[:, self._species_mask] / self._scale
        )
        return out

    def _backward(self, data):
        out = data.clone()
        y = self._clamp_y(data[:, self._species_mask])
        out[:, self._species_mask] = torch.sinh(y) * self._scale / self.a
        return out

    def fit(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")

        col_names = list(x.colName)
        self._col_names = col_names
        self._species_mask = self._build_species_mask(col_names, x.storingDevice)
        self._scale = self._compute_scale(x.data)
        self._y_max = self._compute_y_max(x.data)
        self._device = x.storingDevice
        self._dtype = x.storingDtype
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

    def J(self, x):
        """dy/dx évalué en x (données dans l'espace d'origine)."""
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        N, F = x.data.shape
        J = torch.ones(N, F, device=x.storingDevice, dtype=x.storingDtype)

        col = x.data[:, self._species_mask]
        # d/dx asinh(a x / x_q) = (a / x_q) / sqrt(1 + (a x / x_q)^2)
        u = self.a * col / self._scale
        J[:, self._species_mask] = (self.a / self._scale) / torch.sqrt(1.0 + u**2)

        return J.to(x.storingDevice).to(x.storingDtype)

    def J_1(self, x):
        """dx/dy évalué en y (données dans l'espace transformé).
        Utilise y clampé pour éviter l'overflow de cosh : la valeur reste finie
        et égale à sa valeur au bord hors de l'intervalle (et non la vraie
        dérivée nulle du clamp dur)."""
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        N, F = x.data.shape
        J_inv = torch.ones(N, F, device=x.storingDevice, dtype=x.storingDtype)

        col = self._clamp_y(x.data[:, self._species_mask])
        # d/dy (x_q * sinh(y) / a) = x_q * cosh(y) / a
        J_inv[:, self._species_mask] = torch.cosh(col) * self._scale / self.a

        return J_inv.to(x.storingDevice).to(x.storingDtype)

    def __repr__(self) -> str:
        return (
            f"asinhScaler(q={self.q!r}, a={self.a}, "
            f"cutoff_factor={self.cutoff_factor}, ste={self.ste})"
        )
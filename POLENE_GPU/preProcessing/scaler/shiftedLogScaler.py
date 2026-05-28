from POLENE_GPU.preProcessing.scaler._BaseTransform import _BaseTransform
from POLENE_GPU.dataHandler.serie import serie

import torch


class ShiftedLogScaler(_BaseTransform):
    def __init__(self, min_shift: float = 1.0):
        """
        Scaler logarithmique avec décalage adaptatif.

        Le décalage est calculé à partir des données lors du fit :
          - Si min(data) > 0  : shift = min(data)  → les valeurs restent à leur
                                échelle naturelle, la plus petite devient log(1).
          - Si min(data) <= 0 : shift = -min(data) + min_shift  → tout devient
                                strictement positif avant le log.

        Parameters
        ----------
        min_shift : float
            Valeur minimale garantie après décalage quand les données contiennent
            des zéros ou des valeurs négatives (défaut : 1.0).
        """
        self.min_shift = min_shift

        self._species_mask = None
        self._col_names    = None
        self._device       = None
        self._dtype        = None
        # shift par colonne (shape: [F_species])
        self._shifts       = None
        # colonnes espèces constantes à 0 : on les laisse intactes (shape: [F_species])
        self._zero_mask    = None

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _build_species_mask(self, col_names, device):
        return torch.tensor(
            [name not in self.NON_SPECIES_COLS for name in col_names],
            dtype=torch.bool,
            device=device,
        )

    def _compute_shifts(self, species_data: torch.Tensor) -> torch.Tensor:
        """
        Calcule un vecteur de shifts de taille [F_species].

        Le shift est AJOUTÉ aux données avant le log pour éloigner les valeurs de 0.

        Pour chaque colonne :
          - min_val > 0  →  shift = min_val
                            → la plus petite valeur devient log(2*min_val),
                              on reste à l'échelle naturelle des données.
          - min_val <= 0 →  shift = -min_val + min_shift
                            → le minimum est ramené à min_shift > 0.
        """
        min_vals = species_data.min(dim=0).values          # [F_species]
        positive_shift = min_vals.clone()                  # cas min > 0
        negative_shift = -min_vals + self.min_shift        # cas min <= 0

        shifts = torch.where(min_vals > 0, positive_shift, negative_shift)
        return shifts

    def _sync(self, x):
        self._species_mask = self._species_mask.to(x.storingDevice)
        self._shifts       = self._shifts.to(x.storingDevice).to(x.storingDtype)
        self._zero_mask    = self._zero_mask.to(x.storingDevice)
        self._device = x.storingDevice
        self._dtype  = x.storingDtype

    def _syncTorch(self, x):
        self._species_mask = self._species_mask.to(x.device)
        self._shifts       = self._shifts.to(x.device).to(x.dtype)
        self._zero_mask    = self._zero_mask.to(x.device)
        self._device = x.device
        self._dtype  = x.dtype

    # ------------------------------------------------------------------
    # Forward / Backward
    # ------------------------------------------------------------------

    def _forward(self, data: torch.Tensor) -> torch.Tensor:
        """log(x + shift) pour les colonnes espèces, sauf colonnes constantes à 0."""
        out = data.clone()
        col = data[:, self._species_mask]                        # [N, F_species]
        shifted = col + self._shifts.unsqueeze(0)               # [N, F_species]
        log_vals = shifted.log()
        # Les colonnes zero_mask restent à 0 (log non appliqué)
        log_vals[:, self._zero_mask] = col[:, self._zero_mask]
        out[:, self._species_mask] = log_vals
        return out

    def _backward(self, data: torch.Tensor) -> torch.Tensor:
        """exp(y) - shift pour les colonnes espèces, sauf colonnes constantes à 0."""
        out = data.clone()
        col = data[:, self._species_mask]
        exp_vals = col.exp() - self._shifts.unsqueeze(0)
        # Les colonnes zero_mask restent intactes (exp non appliqué)
        exp_vals[:, self._zero_mask] = col[:, self._zero_mask]
        out[:, self._species_mask] = exp_vals
        return out

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def fit(self, x):
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")

        col_names = list(x.colName)
        self._col_names    = col_names
        self._species_mask = self._build_species_mask(col_names, x.storingDevice)
        self._device = x.storingDevice
        self._dtype  = x.storingDtype

        species_data = x.data[:, self._species_mask]
        self._shifts = self._compute_shifts(species_data).to(x.storingDtype)
        # Colonnes constantes à 0 : max == 0 (et donc min == 0)
        self._zero_mask = species_data.max(dim=0).values == 0
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

    def transformTorch(self, x: torch.Tensor) -> torch.Tensor:
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

    def inverse_transformTorch(self, x: torch.Tensor) -> torch.Tensor:
        self._syncTorch(x)
        return self._backward(x)

    # ------------------------------------------------------------------
    # Jacobiens
    # ------------------------------------------------------------------

    def J(self, x):
        """
        Jacobien diagonale de la transformation forward.
        d/dx [log(x + shift)] = 1 / (x + shift)
        """
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        N, F = x.data.shape
        J = torch.ones(N, F, device=x.storingDevice, dtype=x.storingDtype)

        col     = x.data[:, self._species_mask]
        shifted = col + self._shifts.unsqueeze(0)       # [N, F_species]

        # Sécurité numérique : évite division par zéro si shifted <= 0
        safe = torch.where(shifted > 0, shifted, torch.ones_like(shifted))
        jac_vals = torch.where(shifted > 0, 1.0 / safe, torch.zeros_like(shifted))
        # Colonnes constantes à 0 : transformation identité → jacobien = 1
        jac_vals[:, self._zero_mask] = 1.0
        J[:, self._species_mask] = jac_vals

        return J

    def J_1(self, x):
        """
        Jacobien diagonale de la transformation inverse.
        d/dy [exp(y) - shift] = exp(y)
        """
        if x.data.dim() != 2:
            raise ValueError("Input must be 2D (N, F)")
        if self._species_mask is None:
            raise RuntimeError("Scaler not fitted")

        self._sync(x)

        N, F = x.data.shape
        J_inv = torch.ones(N, F, device=x.storingDevice, dtype=x.storingDtype)
        J_inv[:, self._species_mask] = x.data[:, self._species_mask].exp()

        return J_inv

    def __repr__(self) -> str:
        shift_info = (
            f", shifts={self._shifts.tolist()}" if self._shifts is not None else ""
        )
        return f"logScalerShift(min_shift={self.min_shift}{shift_info})"
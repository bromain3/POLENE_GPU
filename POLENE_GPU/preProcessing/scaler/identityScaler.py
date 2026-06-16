from POLENE_GPU.preProcessing.scaler._BaseTransform import _BaseTransform
from POLENE_GPU.dataHandler.serie import serie

import torch


class identityScaler(_BaseTransform):
    """
    No-op scaler — passes data through unchanged in all directions.

    Used as a default input/output scaler when no scaling is desired,
    e.g. as the default scaler in MixtureOfExperts so that validators
    that call model.inputScaler / model.outputScaler always find a
    valid scaler object.
    """

    def fit(self, x):
        return self

    def fit_transform(self, x):
        return x

    def transform(self, x):
        return x

    def transformTorch(self, x):
        return x

    def inverse_transform(self, x):
        return x

    def inverse_transformTorch(self, x):
        return x

    def J(self, x):
        """Jacobian diagonal = ones  (identity map)."""
        n, f = x.data.shape[0], x.data.shape[1]
        return torch.ones(n, f, device=x.data.device, dtype=x.data.dtype)

    def J_1(self, x):
        """Inverse Jacobian diagonal = ones  (identity map)."""
        return self.J(x)
import matplotlib.pyplot as plt
import numpy as np
import torch

import loss_landscapes
import loss_landscapes.metrics as metrics

from losscape.create_landscape import create_3D_losscape, _set_weights, _reset_weights
from losscape.create_directions import create_random_directions
import losscape.compute_loss as _losscape_compute_loss_module


def _get_params(model):
    return [p.data.clone() for p in model.parameters()]


def _set_params(model, params):
    for p, new_p in zip(model.parameters(), params):
        p.data.copy_(new_p)


def _rand_direction_like(params):
    return [torch.randn_like(p) for p in params]


def _filter_normalize_li2018(direction, params):
    """
    Normalisation exacte de Li et al. 2018 :
    - biais (dim <= 1) : mis à zéro, pas de contribution
    - poids (dim >= 2) : normalisation filtre par filtre (ligne par ligne)
      d_i <- d_i * ||w_i|| / ||d_i||
    """
    normalized = []
    for d, p in zip(direction, params):
        if d.dim() <= 1:
            # Li et al. : biais ignorés (mis à zéro)
            normalized.append(torch.zeros_like(d))
        else:
            # normalisation par filtre = par ligne de la matrice
            d_norm = d.view(d.size(0), -1).norm(dim=1, keepdim=True)
            p_norm = p.view(p.size(0), -1).norm(dim=1, keepdim=True)
            scale = (p_norm / (d_norm + 1e-10)).view(
                p.size(0), *([1] * (d.dim() - 1))
            )
            normalized.append(d * scale)
    return normalized


def _add_perturbation(base_params, dir1, dir2, alpha, beta):
    return [
        b + alpha * d1 + beta * d2
        for b, d1, d2 in zip(base_params, dir1, dir2)
    ]


def random_plane(model, loss_fn, X, Y, distance=1.0, steps=30):
    """Landscape maison, normalisation Li et al. 2018 exacte."""
    model.eval()
    base_params = _get_params(model)

    dir1 = _rand_direction_like(base_params)
    dir2 = _rand_direction_like(base_params)

    dir1 = _filter_normalize_li2018(dir1, base_params)
    dir2 = _filter_normalize_li2018(dir2, base_params)

    coords = torch.linspace(-distance, distance, steps)
    landscape = np.zeros((steps, steps))

    with torch.no_grad():
        for i, alpha in enumerate(coords):
            for j, beta in enumerate(coords):
                perturbed = _add_perturbation(
                    base_params, dir1, dir2, alpha.item(), beta.item()
                )
                _set_params(model, perturbed)
                output = model(model.inputScaler.transform(X))
                loss_val = loss_fn(
                    model.outputScaler.inverse_transform(output).data,
                    Y.data
                )
                landscape[i, j] = loss_val.item()

    _set_params(model, base_params)
    return landscape


def _compute_loss_patched(model, Xsample, Ysample, loss_fn):
    """
    Remplace compute_loss de losscape : bypass le DataLoader/get_batch,
    travaille directement avec nos tenseurs serie.
    """
    with torch.no_grad():
        output = model(Xsample)
        loss = loss_fn(
            model.outputScaler.inverse_transform(output).data,
            Ysample.data
        )
    return loss.item()


def random_plane_losscape(model, loss_fn, Xsample, Ysample, distance=1.0, steps=30):
    """
    Landscape via les directions de losscape (alxndrTL) mais avec
    notre propre boucle pour contourner l'API DataLoader/get_batch.
    Normalisation : layer-wise comme losscape (≠ filter-wise de Li et al.)
    Les biais sont aussi mis à zéro (identique à Li et al.)
    """
    model.eval()
    init_weights = [p.data.clone() for p in model.parameters()]

    directions = create_random_directions(model)

    X_coords = np.linspace(-distance, distance, steps)
    Y_coords = np.linspace(-distance, distance, steps)
    losses = np.empty((steps, steps))

    total = steps * steps
    count = 0

    with torch.no_grad():
        for i, x in enumerate(X_coords):
            for j, y in enumerate(Y_coords):
                _set_weights(model, init_weights, directions, np.array([x, y]))
                losses[i, j] = _compute_loss_patched(model, Xsample, Ysample, loss_fn)
                count += 1
                if count % (total // 10) == 0:
                    print(f"  losscape: {count}/{total} ({100*count//total}%)")

    _reset_weights(model, init_weights)
    return losses


class LossLandscape:
    def __init__(self, model, loss, X, Y, batchSize=2048, computationDevice=None, computationDtype=None):
        self.model = model.to(computationDevice).to(computationDtype)
        Xsample, Ysample = self.__drawSamples(X, Y, batchSize)
        self.Xsample = Xsample.to(computationDevice).to(computationDtype)
        self.Ysample = Ysample.to(computationDevice).to(computationDtype)
        self.loss = loss
        self.computationDevice = computationDevice
        self.computationDtype = computationDtype

    def __drawSamples(self, X, Y, batchSize):
        N = X.nSamples
        indices = torch.arange(N)
        indices = indices[torch.randperm(N)]
        idx = indices[:batchSize]
        return X[idx].contiguous(), Y[idx].contiguous()

    def computeLandscape_hm(self, distance=1.0, discretization=30):
        """Landscape maison, Li et al. 2018 exact (filter norm, biais=0)."""
        return random_plane(
            self.model, self.loss,
            self.Xsample, self.Ysample,
            distance=distance, steps=discretization
        )

    def computeLandscape_losscape(self, distance=1.0, discretization=30):
        """
        Landscape via directions losscape (layer norm, biais=0).
        Boucle maison pour bypasser l'API DataLoader.
        """
        # Pré-calcul : X scalé, Y en unités physiques (inverse transform après prédiction)
        Xsample_scaled = self.model.inputScaler.transform(self.Xsample)
        return random_plane_losscape(
            self.model, self.loss,
            Xsample_scaled, self.Ysample,
            distance=distance, steps=discretization
        )

    def plotLandscape(self, landscape, title=''):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        N = landscape.shape[0]
        M = landscape.shape[1]
        X = np.array([[j for j in range(M)] for _ in range(N)])
        Y = np.array([[i for _ in range(M)] for i in range(N)])
        surf = ax.plot_surface(
            X, Y, np.log10(landscape + 1e-10),
            cmap='coolwarm', edgecolor='none'
        )
        plt.colorbar(surf, ax=ax)
        if title:
            ax.set_title(title)
        plt.tight_layout()
        plt.show()

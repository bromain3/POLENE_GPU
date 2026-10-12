import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from POLENE_GPU.NN.validator._commonValidator import _commonValidator
from POLENE_GPU.NN.models.mixtureOfExperts import MixtureOfExperts


class oneStepPredictionValidator(_commonValidator):
    def validate(self, model, X, Y, tX, showClusters: bool = False):
        self._checkInput(model, X, Y)

        model.eval()
        model.to(self.validationDtype).to(self.validationDevice)
        X = X.clone().to(self.validationDtype).to(self.validationDevice)
        Y = Y.clone().to(self.validationDtype).to(self.validationDevice)

        use_clusters = showClusters and isinstance(model, MixtureOfExperts)

        with torch.no_grad():
            if model.predictState:
                preds = model.outputScaler.inverse_transform(model(model.inputScaler.transform(X)))
            else:
                omega_hat = model(model.inputScaler.transform(X))
                preds = model.output2State(X, omega_hat)

            if use_clusters:
                labels = model.predict_cluster(model.inputScaler.transform(X))
                labels = labels.cpu().numpy()
                n_clusters = model.n_experts
                cmap = cm.get_cmap("tab10", n_clusters)
                cluster_colors = [cmap(k) for k in range(n_clusters)]

        for col in X.colName:
            fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            t      = tX['t'].cpu().detach().numpy()
            pred   = preds[col].cpu().detach().numpy()
            target = Y[col].cpu().detach().numpy()

            # =========================
            # Plot linéaire
            # =========================
            if use_clusters:
                _plot_colored_segments(axs[0], t, pred, labels, cluster_colors, linestyle='--')
            else:
                axs[0].plot(t, pred, label='prediction', color='red', linestyle='--')

            axs[0].plot(t, target, label='target', color='blue')
            axs[0].set_title(f"{col} (linear scale)")
            axs[0].legend()
            axs[0].grid(True)

            # =========================
            # Plot logarithmique
            # =========================
            eps = 1e-30

            if use_clusters:
                _plot_colored_segments(axs[1], t, np.abs(pred) + eps, labels, cluster_colors,
                                       linestyle='--', log=True)
            else:
                axs[1].semilogy(t, np.abs(pred) + eps, label='prediction', color='red', linestyle='--')

            axs[1].semilogy(t, np.abs(target) + eps, label='target', color='blue')
            axs[1].set_title(f"{col} (log scale)")
            axs[1].legend()
            axs[1].grid(True)

            axs[1].set_xlabel("t")

            plt.tight_layout()
            plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: draw a 1-D signal as contiguous same-cluster segments
# ──────────────────────────────────────────────────────────────────────────────

def _plot_colored_segments(ax, t, values, labels, colors, linestyle='--', log=False):
    """
    Draw `values` vs `t` on `ax`, coloring each contiguous run of identical
    cluster labels with the corresponding entry in `colors`.

    A short overlap of one sample between adjacent segments ensures the
    polyline looks continuous (no gap between colour changes).

    Parameters
    ----------
    ax        : matplotlib Axes
    t         : array-like, shape (N,)
    values    : array-like, shape (N,)
    labels    : int array, shape (N,)  — cluster index per sample
    colors    : list of RGBA tuples, length == n_clusters
    linestyle : str
    log       : bool — use semilogy instead of plot
    """
    plot_fn = ax.semilogy if log else ax.plot

    n = len(labels)
    i = 0
    drawn_labels = set()

    while i < n:
        k = labels[i]

        # Find the end of the current run
        j = i + 1
        while j < n and labels[j] == k:
            j += 1

        # Extend one sample into the next segment for visual continuity
        end = min(j, n - 1) + 1

        label_str = f"cluster {k}" if k not in drawn_labels else None
        drawn_labels.add(k)

        plot_fn(t[i:end], values[i:end],
                color=colors[k], linestyle=linestyle, label=label_str, linewidth=4.0)

        i = j
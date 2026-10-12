import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from POLENE_GPU.NN.validator._commonValidator import _commonValidator
from POLENE_GPU.NN.models.mixtureOfExperts import MixtureOfExperts
from POLENE_GPU.dataHandler.serie import concat, serie


class autoregressive0DValidator(_commonValidator):
    def validate(self, model, X, Y, tX, dtcfd, showClusters: bool = False):
        self._checkInput(model, X, Y)

        model.eval()
        model.to(self.validationDtype).to(self.validationDevice)
        X = X.clone().to(self.validationDtype).to(self.validationDevice)
        Y = Y.clone().to(self.validationDtype).to(self.validationDevice)

        use_clusters = showClusters and isinstance(model, MixtureOfExperts)

        preds       = [X[0] if model.predictState else X[0].data]
        time        = [0]
        step_labels = []   # cluster label for each generated step (empty when not used)

        with torch.no_grad():
            while time[-1] < tX['t'][-1].item():
                current = preds[-1].clone()

                if use_clusters:
                    label = model.predict_cluster(
                        model.inputScaler.transformTorch(current.data), tensorInput=True
                    )
                    step_labels.append(int(label[0].item()))

                if model.predictState:
                    next_state = model.outputScaler.inverse_transform(
                        model(model.inputScaler.transform(current))
                    )
                else:
                    omega_hat = model(model.inputScaler.transformTorch(current.data.clone()), tensorInput=True)
                    next_state = model.torchOutput2State(current.data, omega_hat)

                preds.append(next_state)
                time.append(time[-1] + dtcfd)

        # The initial step (X[0]) has no label yet — assign cluster 0 as a neutral default
        # so that len(step_labels) == len(time) when clusters are used.
        if use_clusters:
            step_labels = [step_labels[0]] + step_labels  # replicate first real label for t=0

            n_clusters    = model.n_experts
            cmap          = cm.get_cmap("tab10", n_clusters)
            cluster_colors = [cmap(k) for k in range(n_clusters)]
            labels_arr    = np.array(step_labels)

        if model.predictState:
            preds = concat(preds)
        else:
            preds = torch.concat(preds)
            preds = serie(preds, colName=X.colName, storingDtype=self.validationDtype, storingDevice=self.validationDevice)

        for col in X.colName:
            fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            t      = tX['t'].cpu().detach().numpy()
            pred   = preds[col].cpu().detach().numpy()
            target = X[col].cpu().detach().numpy()

            # =========================
            # Plot linéaire
            # =========================
            if use_clusters:
                _plot_colored_segments(axs[0], time, pred, labels_arr, cluster_colors, linestyle='--')
            else:
                axs[0].plot(time, pred, label='prediction', color='red', linestyle='--')

            axs[0].plot(t, target, label='target', color='blue')
            axs[0].set_title(f"{col} (linear scale)")
            axs[0].legend()
            axs[0].grid(True)

            # =========================
            # Plot logarithmique
            # =========================
            eps = 1e-30

            if use_clusters:
                _plot_colored_segments(axs[1], time, np.abs(pred) + eps, labels_arr,
                                       cluster_colors, linestyle='--', log=True)
            else:
                axs[1].semilogy(time, np.abs(pred) + eps, label='prediction', color='red', linestyle='--')

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

    t      = np.asarray(t)
    values = np.asarray(values)
    n      = len(labels)
    i      = 0
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
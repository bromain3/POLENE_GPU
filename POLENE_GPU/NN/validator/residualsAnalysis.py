import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from POLENE_GPU.NN.validator._commonValidator import _commonValidator
from POLENE_GPU.NN.models.mixtureOfExperts import MixtureOfExperts
from POLENE_GPU.dataHandler.serie import serie


class residualAnalysisValidator(_commonValidator):
    def validate(self, model, X, Y, plot=False, showClusters: bool = False):
        self._checkInput(model, X, Y)

        model.eval()
        model.to(self.validationDtype).to(self.validationDevice)
        X = X.clone().to(self.validationDtype).to(self.validationDevice)
        Y = Y.clone().to(self.validationDtype).to(self.validationDevice)

        use_clusters = showClusters and isinstance(model, MixtureOfExperts)

        with torch.no_grad():
            Y_norm = model.outputScaler.transform(Y)
            Y_hat_norm = model(model.inputScaler.transform(X))  # shape [N, n_out]

            labels = None
            n_clusters = None
            if use_clusters:
                labels = model.predict_cluster(model.inputScaler.transform(X))
                labels = labels.cpu().numpy()
                n_clusters = model.n_experts

        residuals_norm = (Y_hat_norm.data - Y_norm.data).cpu().detach().numpy()      # [N, n_out]
        Y_norm_np      = Y_norm.data.cpu().detach().numpy()
        Y_hat_norm_np  = Y_hat_norm.data.cpu().detach().numpy()


        # TODO: activate those lines after serie object is fixed (implementation of .mean(), etc)
        #perColStats, globalStats = self.computeStats(preds, Y)
        #self.printStats(perColStats, globalStats)

        col_names = Y.colName  # list of output names (species / variables)

        if plot:
            self._plot(col_names, Y_norm_np, Y_hat_norm_np, residuals_norm,
                       labels=labels, n_clusters=n_clusters)
        else:
            self._print_stats(col_names, residuals_norm)

    # ------------------------------------------------------------------
    def _plot(self, col_names, Y_norm, Y_hat_norm, residuals, max_subplots=4,
              labels=None, n_clusters=None):
        n_total = len(col_names)

        use_clusters = labels is not None
        if use_clusters:
            cmap = cm.get_cmap("tab10", n_clusters)
            cluster_colors = [cmap(k) for k in range(n_clusters)]
            point_colors = [cluster_colors[k] for k in labels]

        for start in range(0, n_total, max_subplots):
            end = min(start + max_subplots, n_total)

            cols_chunk = col_names[start:end]
            k = len(cols_chunk)

            # --- figures pour ce chunk
            fig_scatter, axes_s = plt.subplots(1, k, figsize=(4 * k, 4))
            fig_hist, axes_h = plt.subplots(1, k, figsize=(4 * k, 4))

            # quand k == 1 matplotlib ne renvoie pas une liste
            if k == 1:
                axes_s = [axes_s]
                axes_h = [axes_h]

            # --- boucle sur le chunk courant
            for j, col in enumerate(cols_chunk):
                i = start + j

                # scatter : y_true_norm vs residual_norm
                if use_clusters:
                    axes_s[j].scatter(Y_norm[:, i], residuals[:, i], s=4, alpha=0.5,
                                       c=point_colors)
                else:
                    axes_s[j].scatter(Y_norm[:, i], residuals[:, i], s=4, alpha=0.5)
                axes_s[j].axhline(0, color="red", linewidth=0.8, linestyle="--")
                axes_s[j].set_xlabel("y_true (normalized)")
                axes_s[j].set_ylabel("residual (normalized)")
                axes_s[j].set_title(col)

                # histogram des résidus
                if use_clusters:
                    res_by_cluster = [residuals[labels == c, i] for c in range(n_clusters)]
                    axes_h[j].hist(res_by_cluster, bins=50, stacked=True,
                                    color=cluster_colors,
                                    label=[f"cluster {c}" for c in range(n_clusters)],
                                    edgecolor="none")
                else:
                    axes_h[j].hist(residuals[:, i], bins=50, edgecolor="none")
                axes_h[j].axvline(0, color="red", linewidth=0.8, linestyle="--")
                axes_h[j].set_xlabel("residual (normalized)")
                axes_h[j].set_title(col)

            if use_clusters:
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=cluster_colors[c], markersize=6,
                                       label=f"cluster {c}") for c in range(n_clusters)]
                fig_scatter.legend(handles=handles, loc="upper right", ncol=n_clusters)
                axes_h[0].legend(loc="upper right", fontsize=8)

            fig_scatter.suptitle(f"Résidus normalisés — scatter ({start}:{end})", fontsize=12)

            fig_hist.suptitle(f"Résidus normalisés — histogramme ({start}:{end})", fontsize=12)

            fig_scatter.tight_layout()
            fig_hist.tight_layout()

            plt.show()

    # ------------------------------------------------------------------
    def _print_stats(self, col_names, residuals):
        """
        Print per-output statistics in the normalized space, sorted by MSE desc.
        residuals : np.ndarray [N, n_out]
        """
        stats = []
        for i, col in enumerate(col_names):
            r = residuals[:, i]
            mse = float(np.mean(r ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(r)))
            bias = float(np.mean(r))
            std = float(np.std(r))
            stats.append(dict(name=col, mse=mse, rmse=rmse, mae=mae, bias=bias, std=std))

        # Sort by MSE descending
        stats.sort(key=lambda s: s["mse"], reverse=True)

        # Global stats
        global_mse = float(np.mean(residuals ** 2))
        global_rmse = float(np.sqrt(global_mse))
        global_mae = float(np.mean(np.abs(residuals)))

        col_w = max(len(s["name"]) for s in stats) + 2
        header = f"{'Output':<{col_w}} {'MSE':>12} {'RMSE':>12} {'MAE':>12} {'Bias':>12} {'Std':>12}"
        sep = "-" * len(header)

        print("\n" + sep)
        print("  Residual analysis — normalized output space (sorted by MSE ↓)")
        print(sep)
        print(header)
        print(sep)
        for s in stats:
            print(
                f"{s['name']:<{col_w}} "
                f"{s['mse']:>12.6e} "
                f"{s['rmse']:>12.6e} "
                f"{s['mae']:>12.6e} "
                f"{s['bias']:>12.6e} "
                f"{s['std']:>12.6e}"
            )
        print(sep)
        print(
            f"{'GLOBAL':<{col_w}} "
            f"{global_mse:>12.6e} "
            f"{global_rmse:>12.6e} "
            f"{global_mae:>12.6e}"
        )
        print(sep + "\n")
import torch
import matplotlib.pyplot as plt
import numpy as np

from POLENE_GPU.NN.validator._commonValidator import _commonValidator

class oneStepPredictionValidator(_commonValidator):
    def validate(self, model, X, Y, tX):
        self._checkInput(model, X, Y)

        model.eval()
        model.to(self.validationDtype).to(self.validationDevice)
        X = X.to(self.validationDtype).to(self.validationDevice)
        Y = Y.to(self.validationDtype).to(self.validationDevice)

        with torch.no_grad():
            preds = model.outputScaler.inverse_transform(model(model.inputScaler.transform(X)))

        # TODO: activate those lines after serie object is fixed (implementation of .mean(), etc)
        #perColStats, globalStats = self.computeStats(preds, Y)
        #self.printStats(perColStats, globalStats)

        for col in X.colName:
            fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            t = tX['t'].cpu().detach().numpy()
            pred = preds[col].cpu().detach().numpy()
            target = Y[col].cpu().detach().numpy()

            # =========================
            # Plot linéaire
            # =========================
            axs[0].plot(t, pred, label='prediction', color='red', linestyle='--')

            axs[0].plot(t, target, label='target', color='blue')

            axs[0].set_title(f"{col} (linear scale)")
            axs[0].legend()
            axs[0].grid(True)

            # =========================
            # Plot logarithmique
            # =========================
            eps = 1e-30  # évite log(0)

            axs[1].semilogy(t, np.abs(pred) + eps, label='prediction', color='red', linestyle='--')

            axs[1].semilogy(t, np.abs(target) + eps, label='target', color='blue')

            axs[1].set_title(f"{col} (log scale)")
            axs[1].legend()
            axs[1].grid(True)

            axs[1].set_xlabel("t")

            plt.tight_layout()
            plt.show()
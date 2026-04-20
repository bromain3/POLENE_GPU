import torch
import matplotlib.pyplot as plt

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
            plt.plot(tX['t'].cpu().detach().numpy(), preds[col].cpu().detach().numpy(), label='prediction', color='red', linestyle='--')
            plt.plot(tX['t'].cpu().detach().numpy(), Y[col].cpu().detach().numpy(), label='target', color='blue')
            plt.legend()
            plt.title(col)
            plt.show()

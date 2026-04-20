import torch
import matplotlib.pyplot as plt

from POLENE_GPU.NN.validator._commonValidator import _commonValidator
from POLENE_GPU.dataHandler.serie import concat

class autoregressive0DValidator(_commonValidator):
    def validate(self, model, X, Y, tX, dtcfd):
        self._checkInput(model, X, Y)

        model.eval()
        model.to(self.validationDtype).to(self.validationDevice)
        X = X.to(self.validationDtype).to(self.validationDevice)
        Y = Y.to(self.validationDtype).to(self.validationDevice)

        preds = [X[0]]
        time = [0]

        with torch.no_grad():
            while time[-1] < tX['t'][-1].item():
                preds.append(model.outputScaler.inverse_transform(model(model.inputScaler.transform(preds[-1]))))
                time.append(time[-1] + dtcfd)

        preds = concat(preds)

        # TODO: activate those lines after serie object is fixed (implementation of .mean(), etc)
        #perColStats, globalStats = self.computeStats(preds, Y)
        #self.printStats(perColStats, globalStats)

        for col in X.colName:
            plt.plot(time, preds[col].cpu().detach().numpy(), label='prediction', color='red', linestyle='--')
            plt.plot(tX['t'].cpu().detach().numpy(), Y[col].cpu().detach().numpy(), label='target', color='blue')
            plt.legend()
            plt.title(col)
            plt.show()

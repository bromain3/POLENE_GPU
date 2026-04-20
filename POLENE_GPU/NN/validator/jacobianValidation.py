import torch
import matplotlib.pyplot as plt

from POLENE_GPU.NN.validator._commonValidator import _commonValidator
from POLENE_GPU.dataHandler.serie import serie

class jacobianValidation(_commonValidator):
    def validate(self, model, X, Y, tX, reactorModel, dt):
        self._checkInput(model, X, Y)

        model.eval()
        model.to(self.validationDtype).to(self.validationDevice)
        X = X.to(self.validationDtype).to(self.validationDevice)
        Y = Y.to(self.validationDtype).to(self.validationDevice)

        v = Y.data - model.outputScaler.inverse_transform(model(model.inputScaler.transform(X))).data

        jv_true = serie(v - dt * reactorModel.jvp(Y.data, None, v), X.colName, storingDevice=X.storingDevice, storingDtype=X.storingDtype)

        # ── JVP du réseau dans l'espace normalisé ─────────────────────────
        def model_fn(input):
            return model.outputScaler.inverse_transformTorch(model(model.inputScaler.transformTorch(input), tensorInput=True))

        _, Jv_network = torch.func.jvp(
            model_fn,
            (Y.data,),
            (v,)
        )  # (N, d)
        Jv_network = serie(Jv_network, X.colName, storingDevice=X.storingDevice, storingDtype=X.storingDtype)

        # TODO: activate those lines after serie object is fixed (implementation of .mean(), etc)
        #perColStats, globalStats = self.computeStats(preds, Y)
        #self.printStats(perColStats, globalStats)

        for col in X.colName:
            plt.plot(tX['t'].cpu().detach().numpy(), jv_true[col].cpu().detach().numpy(), label='prediction', color='red', linestyle='--')
            plt.plot(tX['t'].cpu().detach().numpy(), Jv_network[col].cpu().detach().numpy(), label='target', color='blue')
            plt.legend()
            plt.title(col)
            plt.show()

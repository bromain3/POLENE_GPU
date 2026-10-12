import torch.nn as nn
import torch
import pickle
import os

from POLENE_GPU.dataHandler.serie import serie


class _wrapper(nn.Module):
    def __init__(self, predictState=True, chemScaler=None, chemStdScaler=None):
        nn.Module.__init__(self)
        self.inputScaler = None
        self.outputScaler = None
        self.predictState = predictState
        self.chemScaler = chemScaler
        self.chemStdScaler = chemStdScaler

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, weights_only=False)
        model = cls._from_config(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.inputScaler  = pickle.loads(checkpoint["scalers"]["input"])
        model.outputScaler = pickle.loads(checkpoint["scalers"]["output"])
        return model

    @classmethod
    def _from_config(cls, config):
        """À surcharger : reconstruit une instance depuis la config."""
        raise NotImplementedError()

    def _get_config(self):
        """À surcharger dans les sous-classes pour retourner les hyperparamètres."""
        raise NotImplementedError()

    def torchForward(self, x):
        raise NotImplementedError()

    def save(self, dir, file):
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, file)
        checkpoint = {
            "state_dict": self.state_dict(),
            "config":     self._get_config(),
            "scalers": {
                "input":  pickle.dumps(self.inputScaler),
                "output": pickle.dumps(self.outputScaler),
            },
        }
        torch.save(checkpoint, path)

    def forward(self, serie, tensorInput=False):
        if not tensorInput:
            x = self.torchForward(serie.data)
            serie.data = x
            return serie
        else:
            return self.torchForward(serie)

    def torchOutput2State(self, x, omega_hat):
        if self.predictState:
            raise NotImplementedError('The model has been trained to predict scaled states. torchOutput2State is thus undefined')
        else:
            omega = self.outputScaler.inverse_transformTorch(omega_hat)
            chemScaledY = self.chemScaler.transformTorch(x) + omega
            #print(chemScaledY.min().item(), -1.0 / self.chemScaler.lambda_)  # si min < -10 → confirmé
            return(self.chemScaler.inverse_transformTorch(chemScaledY))

    def output2State(self, x, omega_hat, tensorInput=False):
        if not tensorInput:
            y = self.torchOutput2State(x.data, omega_hat.data)
            return serie(y, colName=omega_hat.colName, storingDevice=omega_hat.storingDevice, storingDtype=omega_hat.storingDtype)
        else:
            return self.torchOutput2State(x, omega_hat)

    def torchState2Omega(self, x, y, dtype=None):
        """omega = chem(y) - chem(x)  (x, y physiques). Inverse de torchOutput2State."""
        if self.chemScaler is None:
            raise ValueError("chemScaler is required to compute omega")
        cx = self.chemScaler.transformTorch(x)
        cy = self.chemScaler.transformTorch(y)
        if dtype is not None:
            cx, cy = cx.to(dtype), cy.to(dtype)
        return cy - cx

class IdentityNetwork(_wrapper):
    def __init__(self):
        nn.Module.__init__(self)
        self.dummy = nn.Parameter(torch.zeros(1))

    def torchForward(self, x):
        return x + 0.0 * self.dummy
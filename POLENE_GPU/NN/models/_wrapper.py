import torch.nn as nn
import torch
import pickle
import os



class _wrapper(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.inputScaler = None
        self.outputScaler = None

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

class IdentityNetwork(_wrapper):
    def __init__(self):
        nn.Module.__init__(self)
        self.dummy = nn.Parameter(torch.zeros(1))

    def torchForward(self, x):
        return x + 0.0 * self.dummy
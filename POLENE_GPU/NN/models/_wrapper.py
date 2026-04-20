import torch.nn as nn

class _wrapper(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.inputScaler = None
        self.outputScaler = None

    def torchForward(self, x):
        raise NotImplementedError()

    def forward(self, serie, tensorInput=False):
        if not tensorInput:
            x = self.torchForward(serie.data)
            serie.data = x
            return serie
        else:
            return self.torchForward(serie)
from POLENE_GPU.NN.models._wrapper import _wrapper

import torch.nn as nn
import torch.nn.functional as F

class _hiddenLayer(nn.Module):
    def __init__(self, hidden_dim, layerNorm=True):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.useLayerNorm = layerNorm

        if self.useLayerNorm:
            self.net = nn.Sequential(
                self.linear,
                nn.LayerNorm(hidden_dim),
                self.act
                )
        else:
            self.net = nn.Sequential(
                self.linear,
                self.act
                )

    def forward(self, x):
        return(self.net(x))

class _MLP(nn.Module):
    def __init__(self, nb_layers, hidden_dim, layerNorm=True):
        super().__init__()
        self.network = nn.Sequential(*[
            _hiddenLayer(hidden_dim, layerNorm) for _ in range(nb_layers)
        ])

    def forward(self, x):
        return(self.network(x))

class MLP(_wrapper):
    def __init__(self, input_dim, nb_layers, hidden_dim, layerNorm=True):
        nn.Module.__init__(self)
        _wrapper.__init__(self)

        self.input = nn.Linear(input_dim, hidden_dim)
        self.input_act = nn.SiLU()
        self.core = _MLP(nb_layers, hidden_dim, layerNorm)
        self.output = nn.Linear(hidden_dim, input_dim)

        if layerNorm:
            self.net = nn.Sequential(
                self.input,
                nn.LayerNorm(hidden_dim),
                self.input_act,
                self.core,
                self.output
            )
        else:
            self.net = nn.Sequential(
                self.input,
                self.input_act,
                self.core,
                self.output
            )

        #self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def torchForward(self, x):
        return(self.net(x))
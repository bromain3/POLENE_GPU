from POLENE_GPU.NN.models._wrapper import _wrapper

import torch.nn as nn


class _hiddenLayer(nn.Module):
    def __init__(self, input_dim, output_dim, layerNorm=True):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.act = nn.SiLU()

        if layerNorm:
            self.net = nn.Sequential(
                self.linear,
                nn.LayerNorm(output_dim),
                self.act
            )
        else:
            self.net = nn.Sequential(
                self.linear,
                self.act
            )

    def forward(self, x):
        return self.net(x)


class _DNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, layerNorm=True):
        super().__init__()

        layers = []

        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(
                _hiddenLayer(
                    previous_dim,
                    hidden_dim,
                    layerNorm
                )
            )
            previous_dim = hidden_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DNN6HL(_wrapper):

    def __init__(self, input_dim=53, layerNorm=False, predictState=True, chemScaler=None, chemStdScaler=None):
        nn.Module.__init__(self)
        _wrapper.__init__(self, predictState, chemScaler, chemStdScaler)

        hidden_dims = [
            200,
            400,
            600,
            600,
            400,
            200
        ]

        self.core = _DNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            layerNorm=layerNorm
        )

        self.output = nn.Linear(200, input_dim)

        self.net = nn.Sequential(
            self.core,
            self.output
        )

    def torchForward(self, x):
        return self.net(x)
from POLENE_GPU.NN.models._wrapper import _wrapper

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, w0=1.0, is_first=False):
        super().__init__()

        self.in_dim = in_dim
        self.w0 = w0
        self.is_first = is_first

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = Sine(w0)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():

            if self.is_first:
                bound = 1 / self.in_dim
            else:
                bound = math.sqrt(6 / self.in_dim) / self.w0

            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return self.activation(self.linear(x))

class SIREN(_wrapper):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        nb_layers,
        w0_initial=30.0,
        w0=1.0
    ):
        super().__init__()

        layers = []

        # Première couche
        layers.append(
            SirenLayer(
                input_dim,
                hidden_dim,
                w0=w0_initial,
                is_first=True
            )
        )

        # Couches internes
        for _ in range(nb_layers - 1):
            layers.append(
                SirenLayer(
                    hidden_dim,
                    hidden_dim,
                    w0=w0
                )
            )

        self.core = nn.Sequential(*layers)

        self.output = nn.Linear(hidden_dim, input_dim)

    def torchForward(self, x):
        return self.output(self.core(x))
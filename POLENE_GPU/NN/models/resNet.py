from POLENE_GPU.NN.models._wrapper import _wrapper

import torch.nn as nn
import torch.nn.functional as F

class _ResBlock(nn.Module):
    def __init__(self, features_dim, hidden_dim, layerNorm):
        super().__init__()
        self.layerNorm = layerNorm
        if layerNorm:
            self.norm = nn.LayerNorm(features_dim)
        self.linear1 = nn.Linear(features_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, features_dim)

    def forward(self, x):
        residual = x
        if self.layerNorm:
            x = self.norm(x)
        x = F.silu(self.linear1(x))
        x = self.linear2(x)
        return x + residual

class _ResNet(nn.Module):
    def __init__(self, features_dim, nb_blocks, hidden_dim, layerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            _ResBlock(features_dim, hidden_dim, layerNorm=layerNorm) for _ in range(nb_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ResNet(_wrapper):
    def __init__(self, input_dim, backbone_dim, nb_blocks, hidden_dim, layerNorm=False):
        nn.Module.__init__(self)
        _wrapper.__init__(self)

        self.bck = backbone_dim
        self.nb_blocks = nb_blocks
        self.dim = hidden_dim

        self.input_dim = input_dim
        self.backbone = nn.Linear(input_dim, backbone_dim)
        self.resnet = _ResNet(backbone_dim, nb_blocks, hidden_dim, layerNorm=layerNorm)
        self.output_layer = nn.Linear(backbone_dim, input_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def torchForward(self, x):
        x = F.silu(self.backbone(x))
        x = self.resnet(x)
        #x = F.silu(x)
        x = self.output_layer(x)
        return x
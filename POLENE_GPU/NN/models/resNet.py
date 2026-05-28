from POLENE_GPU.NN.models._wrapper import _wrapper

import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, features_dim, hidden_dim, layerNorm, activationOnOutput=False):
        super().__init__()
        self.layerNorm = layerNorm
        self.activationOnOutput = activationOnOutput

        self._norm    = nn.LayerNorm(features_dim) if layerNorm else nn.Identity()
        self.linear1  = nn.Linear(features_dim, hidden_dim)
        self.act1     = nn.SiLU()   # ← module nommé : hookable
        self.linear2  = nn.Linear(hidden_dim, features_dim)
        self.act_out  = nn.SiLU() if activationOnOutput else nn.Identity()

    def forward(self, x):
        residual = self._norm(x)
        residual = self.act1(self.linear1(residual))
        residual = self.linear2(residual)
        return self.act_out(x + residual)


class _ResNet(nn.Module):
    def __init__(self, features_dim, nb_blocks, hidden_dim, layerNorm, activationOnOutput=False):
        super().__init__()
        self.blocks = nn.Sequential(*[
            _ResBlock(
                features_dim, hidden_dim,
                layerNorm=layerNorm,
                activationOnOutput=activationOnOutput,
            )
            for _ in range(nb_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)


class ResNet(_wrapper):
    def __init__(
        self,
        input_dim,
        backbone_dim,
        nb_blocks,
        hidden_dim,
        layerNorm=True,
        useBackbone=True,
        kaimigInit=False,
        activationOnOutput=False,
    ):
        nn.Module.__init__(self)
        _wrapper.__init__(self)

        self.bck              = backbone_dim
        self.nb_blocks        = nb_blocks
        self.dim              = hidden_dim
        self.input_dim        = input_dim
        self.useBackbone      = useBackbone
        self.layerNorm        = layerNorm
        self.activationOnOutput = activationOnOutput

        if useBackbone:
            self.backbone     = nn.Linear(input_dim, backbone_dim)
            self.backbone_act = nn.SiLU()   # ← module nommé : hookable
            self.resnet       = _ResNet(backbone_dim, nb_blocks, hidden_dim,
                                        layerNorm=layerNorm,
                                        activationOnOutput=activationOnOutput)
            self.output_layer = nn.Linear(backbone_dim, input_dim)
            # net reste utilisé pour le forward — on garde la séquence lisible
            self.net = nn.Sequential(
                self.backbone,
                self.backbone_act,
                self.resnet,
                self.output_layer,
            )
        else:
            backbone_dim  = input_dim
            self.resnet   = _ResNet(backbone_dim, nb_blocks, hidden_dim,
                                    layerNorm=layerNorm,
                                    activationOnOutput=activationOnOutput)
            self.net      = self.resnet

        if kaimigInit:
            self.kaimig_init_weights()

    def _get_config(self):
        return {
            "input_dim":          self.input_dim,
            "backbone_dim":       self.bck,
            "nb_blocks":          self.nb_blocks,
            "hidden_dim":         self.dim,
            "layerNorm":          self.layerNorm,
            "useBackbone":        self.useBackbone,
            "activationOnOutput": self.activationOnOutput,
        }

    @classmethod
    def _from_config(cls, config):
        return cls(**config)

    def kaimig_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def enableFineTuning(self, freeze_backbone=False, freeze_resnet=False, freeze_output=False):
        if freeze_backbone and hasattr(self, "backbone"):
            for p in self.backbone.parameters():
                p.requires_grad = False

        if freeze_resnet:
            for p in self.resnet.parameters():
                p.requires_grad = False

        if freeze_output and hasattr(self, "output_layer"):
            for p in self.output_layer.parameters():
                p.requires_grad = False

    def disableFineTuning(self):
        if self.useBackbone:
            for p in self.backbone.parameters():
                p.requires_grad = True
            for p in self.output_layer.parameters():
                p.requires_grad = True
        for p in self.resnet.parameters():
            p.requires_grad = True

    def torchForward(self, x):
        return self.net(x)
from POLENE_GPU.NN.models._wrapper import _wrapper

import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, features_dim, hidden_dim, layerNorm):
        super().__init__()
        self.layerNorm = layerNorm

        self._norm    = nn.LayerNorm(features_dim) if layerNorm else nn.Identity()
        self.linear1  = nn.Linear(features_dim, hidden_dim)
        self.act1     = nn.SiLU()   # ← module nommé : hookable
        self.linear2  = nn.Linear(hidden_dim, features_dim)

    def forward(self, x):
        residual = self._norm(x)
        residual = self.act1(self.linear1(residual))
        residual = self.linear2(residual)
        return x + residual


class _ResNet(nn.Module):
    def __init__(self, features_dim, nb_blocks, hidden_dim, layerNorm):
        super().__init__()
        self.blocks = nn.Sequential(*[
            _ResBlock(
                features_dim, hidden_dim,
                layerNorm=layerNorm,
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
    ):
        nn.Module.__init__(self)
        _wrapper.__init__(self)

        self.bck              = backbone_dim
        self.nb_blocks        = nb_blocks
        self.dim              = hidden_dim
        self.input_dim        = input_dim
        self.useBackbone      = useBackbone
        self.layerNorm        = layerNorm

        if useBackbone:
            self.backbone     = nn.Linear(input_dim, backbone_dim)
            self.backbone_act = nn.SiLU()   # ← module nommé : hookable
            self.resnet       = _ResNet(backbone_dim, nb_blocks, hidden_dim, layerNorm=layerNorm)
            self.resnet_act   = nn.SiLU()   # ← module nommé : hookable
            self.output_layer = nn.Linear(backbone_dim, input_dim)
            # net reste utilisé pour le forward — on garde la séquence lisible
            if self.layerNorm:
                self.net = nn.Sequential(
                    self.backbone,
                    #nn.LayerNorm(backbone_dim),
                    self.backbone_act,
                    self.resnet,
                    #nn.LayerNorm(backbone_dim),
                    self.resnet_act,
                    self.output_layer,
                )
            else:
                self.net = nn.Sequential(
                    self.backbone,
                    self.backbone_act,
                    self.resnet,
                    self.resnet_act,
                    self.output_layer,
                )
        else:
            backbone_dim  = input_dim
            self.resnet   = _ResNet(backbone_dim, nb_blocks, hidden_dim, layerNorm=layerNorm)
            self.net      = self.resnet

        #self._initialize_weights()

    def _initialize_weights(self):
        # 1. On applique d'abord l'initialisation de base (Kaiming) à tout le réseau
        self.apply(self._base_init)

        # 2. On applique le "Zero-Init" spécifiquement aux blocs résiduels
        for m in self.modules():
            if isinstance(m, _ResBlock):
                # On force la dernière couche linéaire du bloc à 0
                nn.init.constant_(m.linear2.weight, 0)
                if m.linear2.bias is not None:
                    nn.init.constant_(m.linear2.bias, 0)

    def _base_init(self, m):
        """ Fonction utilitaire appelée par self.apply() """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


    def _get_config(self):
        return {
            "input_dim":          self.input_dim,
            "backbone_dim":       self.bck,
            "nb_blocks":          self.nb_blocks,
            "hidden_dim":         self.dim,
            "layerNorm":          self.layerNorm,
            "useBackbone":        self.useBackbone,
        }

    @classmethod
    def _from_config(cls, config):
        return cls(**config)

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
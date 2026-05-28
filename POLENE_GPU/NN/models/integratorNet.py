from POLENE_GPU.NN.models._wrapper import _wrapper

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    """MLP générique : nb_layers couches cachées de taille hidden_dim."""
    def __init__(self, input_dim, output_dim, nb_layers, hidden_dim, layerNorm=False, activationOnOutput=True):
        super().__init__()
        self.activationOnOutput = activationOnOutput

        layers = []
        in_dim = input_dim
        for _ in range(nb_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if layerNorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._out_act = F.silu if activationOnOutput else (lambda x: x)

    def forward(self, x):
        return self._out_act(self.net(x))


class IntegratorNet(_wrapper):
    """
    Architecture inspirée d'un intégrateur à deux branches :

        x ──────────────────────────────────────────► [identity branch]
         │                                                     │
         └──► [source_branch : MLP(input_dim → source_dim)] ──┤
                                                               │
                                                    concat(x, source)
                                                               │
                                                    [integrator : MLP → input_dim]
                                                               │
                                                             output

    Args:
        input_dim      : taille de x
        source_dim     : taille de la sortie de la branche source
        source_layers  : profondeur du MLP source (nb couches cachées)
        source_width   : largeur du MLP source
        integrator_layers : profondeur du MLP intégrateur
        integrator_width  : largeur du MLP intégrateur
        layerNorm      : active LayerNorm dans les deux MLPs
        activationOnOutput : applique SiLU sur la sortie finale
        kaimingInit    : initialisation Kaiming
    """

    def __init__(
        self,
        input_dim,
        source_dim,
        source_layers,
        source_width,
        integrator_layers,
        integrator_width,
        layerNorm=False,
        activationOnOutput=False,
        kaimingInit=False,
    ):
        nn.Module.__init__(self)
        _wrapper.__init__(self)

        self.input_dim         = input_dim
        self.source_dim        = source_dim
        self.source_layers     = source_layers
        self.source_width      = source_width
        self.integrator_layers = integrator_layers
        self.integrator_width  = integrator_width
        self.layerNorm         = layerNorm
        self.activationOnOutput = activationOnOutput

        # Branche source : x → termes sources
        self.source_branch = _MLP(
            input_dim=input_dim,
            output_dim=source_dim,
            nb_layers=source_layers,
            hidden_dim=source_width,
            layerNorm=layerNorm,
            activationOnOutput=True,   # activation interne toujours active
        )

        # Branche intégrateur : concat(x, source) → output de taille input_dim
        self.integrator = _MLP(
            input_dim=input_dim + source_dim,
            output_dim=input_dim,
            nb_layers=integrator_layers,
            hidden_dim=integrator_width,
            layerNorm=layerNorm,
            activationOnOutput=activationOnOutput,
        )

        if kaimingInit:
            self.kaiming_init_weights()

    def _get_config(self):
        return {
            "input_dim":          self.input_dim,
            "source_dim":         self.source_dim,
            "source_layers":      self.source_layers,
            "source_width":       self.source_width,
            "integrator_layers":  self.integrator_layers,
            "integrator_width":   self.integrator_width,
            "layerNorm":          self.layerNorm,
            "activationOnOutput": self.activationOnOutput,
        }

    @classmethod
    def _from_config(cls, config):
        return cls(**config)

    def kaiming_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def enableFineTuning(self, freeze_source=False, freeze_integrator=False):
        if freeze_source:
            for p in self.source_branch.parameters():
                p.requires_grad = False
        if freeze_integrator:
            for p in self.integrator.parameters():
                p.requires_grad = False

    def disableFineTuning(self):
        for p in self.parameters():
            p.requires_grad = True

    def torchForward(self, x):
        source = self.source_branch(x)            # branche source : f(x)
        fused  = torch.cat([x, source], dim=-1)   # identité + source
        return self.integrator(fused)             # "intégration"
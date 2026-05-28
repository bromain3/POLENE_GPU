"""
TransformerEncoder — chaque feature est un token avec son propre embedding appris.
Architecture : embed(x_i) → TransformerEncoder → tête de régression par token → X_t+dt

API compatible avec _wrapper (torchForward / _get_config / _from_config).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from POLENE_GPU.NN.models._wrapper import _wrapper


# ---------------------------------------------------------------------------
# Fonctions d'activation disponibles
# ---------------------------------------------------------------------------
ACTIVATIONS = {
    "silu":  F.silu,
    "relu":  F.relu,
    "gelu":  F.gelu,
    "tanh":  torch.tanh,
    "none":  lambda x: x,
}


# ---------------------------------------------------------------------------
# Embedding par feature : scalaire → ℝ^d_model
# Chaque feature i dispose de sa propre instance (poids non partagés).
# ---------------------------------------------------------------------------
class _FeatureEmbedding(nn.Module):
    """
    Encode un scalaire x_i vers ℝ^d_model.

    embedding_type :
      "linear"  — projection affine  : d_model poids par feature
      "mlp"     — MLP 1 couche cachée : plus expressif, capture non-linéarités
      "fourier" — features de Fourier aléatoires puis projection (bon pour
                  variables périodiques ou à grande dynamique)
    """
    def __init__(self, d_model: int, embedding_type: str = "mlp",
                 embed_hidden: int = 32, activation: str = "silu"):
        super().__init__()
        self.embedding_type = embedding_type
        self.act = ACTIVATIONS[activation]

        if embedding_type == "linear":
            # scalaire (1) → d_model
            self.net = nn.Linear(1, d_model)

        elif embedding_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(1, embed_hidden),
                nn.SiLU(),                        # activation fixe dans le MLP interne
                nn.Linear(embed_hidden, d_model),
            )

        elif embedding_type == "fourier":
            # Projection Fourier aléatoire (non entraînée) + couche linéaire apprise
            B = torch.randn(1, embed_hidden) * 10.0   # fréquences aléatoires
            self.register_buffer("B", B)
            # cos + sin → 2*embed_hidden → d_model
            self.proj = nn.Linear(2 * embed_hidden, d_model)

        else:
            raise ValueError(f"embedding_type inconnu : {embedding_type!r}. "
                             f"Choix : 'linear', 'mlp', 'fourier'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch,) → (batch, d_model)"""
        x = x.unsqueeze(-1)                       # (batch, 1)
        if self.embedding_type == "fourier":
            proj = x @ self.B                     # (batch, embed_hidden)
            x = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
            return self.act(self.proj(x))
        return self.net(x)                        # linear / mlp gèrent seuls l'activation


# ---------------------------------------------------------------------------
# Tête de régression par token : ℝ^d_model → ℝ^1
# ---------------------------------------------------------------------------
class _RegressionHead(nn.Module):
    """
    Projette le vecteur contextuel d'un token vers un scalaire de sortie.

    head_type :
      "linear"  — simple couche linéaire
      "mlp"     — MLP 1 couche cachée
    """
    def __init__(self, d_model: int, head_type: str = "mlp",
                 head_hidden: int = 32, activation: str = "silu"):
        super().__init__()
        self.act = ACTIVATIONS[activation]
        if head_type == "linear":
            self.net = nn.Linear(d_model, 1)
        elif head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.SiLU(),
                nn.Linear(head_hidden, 1),
            )
        else:
            raise ValueError(f"head_type inconnu : {head_type!r}. Choix : 'linear', 'mlp'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, d_model) → (batch,)"""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Architecture principale
# ---------------------------------------------------------------------------
class TransformerEncoderModel(_wrapper):
    """
    Transformer-Encoder où chaque feature est un token.

    Paramètres
    ----------
    n_features       : nombre de features en entrée (= nombre de tokens = sortie)
    d_model          : dimension de l'espace des embeddings / attention
    nhead            : nombre de têtes d'attention (doit diviser d_model)
    num_layers       : nombre de couches TransformerEncoderLayer
    dim_feedforward  : dimension du FFN interne à chaque couche
    dropout          : taux de dropout (0.0 pour désactiver)
    norm_first       : Pre-LN (True, plus stable) ou Post-LN (False, original)
    embedding_type   : 'linear' | 'mlp' | 'fourier'
    embed_hidden     : taille de la couche cachée des embeddings (si mlp/fourier)
    embed_activation : activation dans les embeddings
    head_type        : 'linear' | 'mlp'
    head_hidden      : taille de la couche cachée des têtes de régression
    head_activation  : activation dans les têtes
    residual         : si True → sortie = x + réseau(x)  (prédit ΔX)
    """

    def __init__(
        self,
        n_features:       int,
        d_model:          int   = 64,
        nhead:            int   = 4,
        num_layers:       int   = 3,
        dim_feedforward:  int   = 256,
        dropout:          float = 0.0,
        norm_first:       bool  = True,
        embedding_type:   str   = "mlp",
        embed_hidden:     int   = 32,
        embed_activation: str   = "silu",
        head_type:        str   = "mlp",
        head_hidden:      int   = 32,
        head_activation:  str   = "silu",
        residual:         bool  = False,
        head_mode:        str   = "per_token",  # "per_token" | "global"

    ):
        nn.Module.__init__(self)
        _wrapper.__init__(self)

        # — stockage des hyperparamètres pour _get_config —
        self.n_features      = n_features
        self.d_model         = d_model
        self.nhead           = nhead
        self.num_layers      = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout         = dropout
        self.norm_first      = norm_first
        self.embedding_type  = embedding_type
        self.embed_hidden    = embed_hidden
        self.embed_activation = embed_activation
        self.head_type       = head_type
        self.head_hidden     = head_hidden
        self.head_activation = head_activation
        self.residual        = residual
        self.head_mode       = head_mode

        # — un embedding distinct par feature —
        self.embeddings = nn.ModuleList([
            _FeatureEmbedding(d_model, embedding_type, embed_hidden, embed_activation)
            for _ in range(n_features)
        ])

        # — Transformer Encoder —
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,       # (batch, seq, d_model) — API moderne
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if norm_first else None,
        )

        if head_mode == "per_token":
            # un head par token — inchangé
            self.heads = nn.ModuleList([
                _RegressionHead(d_model, head_type, head_hidden, head_activation)
                for _ in range(n_features)
            ])
        elif head_mode == "global":
            # une seule tête partagée appliquée en broadcast sur (B, n_features, d_model)
            if head_type == "linear":
                self.heads = nn.Linear(d_model, 1)
            elif head_type == "mlp":
                self.heads = nn.Sequential(
                    nn.Linear(d_model, head_hidden),
                    nn.SiLU(),
                    nn.Linear(head_hidden, 1),
                )
            else:
                raise ValueError(f"head_type inconnu : {head_type!r}")
        else:
            raise ValueError(f"head_mode inconnu : {head_mode!r}. Choix : 'per_token', 'global'")

    # -----------------------------------------------------------------------
    def torchForward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, n_features)  →  x_next : (batch, n_features)
        """
        batch = x.shape[0]

        # 1. Embedding : chaque feature → vecteur d_model
        #    On empile pour obtenir (batch, n_features, d_model)
        tokens = torch.stack(
            [self.embeddings[i](x[:, i]) for i in range(self.n_features)],
            dim=1
        )  # (batch, n_features, d_model)

        # 2. Transformer Encoder — attention bidirectionnelle entre tous les tokens
        context = self.transformer(tokens)  # (batch, n_features, d_model)

        # 3. Tête de régression par token → scalaire
        if self.head_mode == "per_token":
            out = torch.stack(
                [self.heads[i](context[:, i, :]) for i in range(self.n_features)],
                dim=1
            )  # (B, n_features)
        else:  # "global"
            out = self.heads(context).squeeze(-1)  # (B, n_features, d_model) → (B, n_features)

        if self.residual:
            return x + out
        return out

    # -----------------------------------------------------------------------
    def _get_config(self) -> dict:
        return {
            "n_features":       self.n_features,
            "d_model":          self.d_model,
            "nhead":            self.nhead,
            "num_layers":       self.num_layers,
            "dim_feedforward":  self.dim_feedforward,
            "dropout":          self.dropout,
            "norm_first":       self.norm_first,
            "embedding_type":   self.embedding_type,
            "embed_hidden":     self.embed_hidden,
            "embed_activation": self.embed_activation,
            "head_type":        self.head_type,
            "head_hidden":      self.head_hidden,
            "head_activation":  self.head_activation,
            "residual":         self.residual,
            "head_mode":        self.head_mode,
        }

    @classmethod
    def _from_config(cls, config: dict) -> "TransformerEncoderModel":
        return cls(**config)

    # -----------------------------------------------------------------------
    def enableFineTuning(
        self,
        freeze_embeddings: bool = False,
        freeze_transformer: bool = False,
        freeze_heads: bool = False,
    ):
        if freeze_embeddings:
            for p in self.embeddings.parameters():
                p.requires_grad = False
        if freeze_transformer:
            for p in self.transformer.parameters():
                p.requires_grad = False
        if freeze_heads:
            for p in self.heads.parameters():
                p.requires_grad = False

    def disableFineTuning(self):
        for p in self.parameters():
            p.requires_grad = True
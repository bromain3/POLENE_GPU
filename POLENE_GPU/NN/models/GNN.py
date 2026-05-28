from POLENE_GPU.NN.models._wrapper import _wrapper

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Couche d'embedding : scalaire → vecteur de dim embed_dim
# ─────────────────────────────────────────────────────────────────────────────
class _NodeEmbedding(nn.Module):
    """
    Projette chaque scalaire xᵢ en un vecteur de dimension embed_dim.
    Entrée  : (batch, n_nodes)
    Sortie  : (batch, n_nodes, embed_dim)
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Linear(1, embed_dim)

    def forward(self, x):
        return self.proj(x.unsqueeze(-1))   # (B, N, 1) → (B, N, E)


# ─────────────────────────────────────────────────────────────────────────────
# Structure du graphe apprise : matrice d'adjacence continue
# ─────────────────────────────────────────────────────────────────────────────
class _LearnedGraph(nn.Module):
    """
    Apprend une matrice d'adjacence A de taille (n_nodes, n_nodes) sans
    aucune information préalable.

    A est paramétrisée par une matrice de logits W brute, puis normalisée
    par softmax sur les voisins (ligne par ligne) pour que les poids soient
    positifs et somment à 1 par nœud.

    On masque la diagonale pour qu'un nœud ne s'agrège pas lui-même
    (la connexion résiduelle s'en charge).
    """
    def __init__(self, n_nodes):
        super().__init__()
        self.n_nodes = n_nodes
        # Logits bruts : initialisés aléatoirement, entièrement appris
        self.logits = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.01)
        # Masque diagonal fixe (pas un paramètre)
        self.register_buffer('diag_mask', ~torch.eye(n_nodes, dtype=torch.bool))

    def forward(self):
        """
        Retourne A : (n_nodes, n_nodes), float, somme par ligne = 1,
        diagonale = 0.
        """
        # Mettre -inf sur la diagonale avant le softmax
        logits_masked = self.logits.masked_fill(~self.diag_mask, float('-inf'))
        return F.softmax(logits_masked, dim=-1)   # (N, N)

    def get_adjacency(self):
        """Alias pratique pour inspecter le graphe appris après entraînement."""
        with torch.no_grad():
            return self.forward()


# ─────────────────────────────────────────────────────────────────────────────
# Une couche de message passing
# ─────────────────────────────────────────────────────────────────────────────
class _GNNLayer(nn.Module):
    """
    Une couche de message passing avec connexion résiduelle et LayerNorm.

    Étapes :
        1. Agrégation  : m_i = sum_j A[i,j] * h_j         (pondération apprise)
        2. Combinaison : u_i = MLP(concat(h_i, m_i))
        3. Résidu      : h_i' = LayerNorm(h_i + u_i)

    Entrée  : (batch, n_nodes, embed_dim)
    Sortie  : (batch, n_nodes, embed_dim)
    """
    def __init__(self, embed_dim, hidden_dim, layerNorm=True):
        super().__init__()
        # MLP de mise à jour : concat(hᵢ, mᵢ) → embed_dim
        self.update_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim) if layerNorm else nn.Identity()

    def forward(self, h, A):
        """
        h : (batch, n_nodes, embed_dim)
        A : (n_nodes, n_nodes)  — matrice d'adjacence normalisée
        """
        # Agrégation : pour chaque nœud i, somme pondérée des voisins j
        # m[b, i, :] = sum_j A[i,j] * h[b, j, :]
        m = torch.einsum('ij, bjd -> bid', A, h)    # (B, N, E)

        # Mise à jour avec connexion résiduelle
        u = self.update_mlp(torch.cat([h, m], dim=-1))   # (B, N, E)
        return self.norm(h + u)


# ─────────────────────────────────────────────────────────────────────────────
# Empilement de couches GNN
# ─────────────────────────────────────────────────────────────────────────────
class _GNNStack(nn.Module):
    def __init__(self, n_layers, embed_dim, hidden_dim, layerNorm=True):
        super().__init__()
        self.layers = nn.ModuleList([
            _GNNLayer(embed_dim, hidden_dim, layerNorm=layerNorm)
            for _ in range(n_layers)
        ])

    def forward(self, h, A):
        for layer in self.layers:
            h = layer(h, A)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Modèle principal — interface identique à ResNet
# ─────────────────────────────────────────────────────────────────────────────
class LearnedGraphNet(_wrapper):
    """
    GNN autoregressif sur un vecteur d'état de taille input_dim.

    Chaque scalaire du vecteur est un nœud du graphe.
    Le graphe (matrice d'adjacence) est entièrement appris, sans
    information préalable.

    Paramètres
    ----------
    input_dim   : taille du vecteur d'état (ex : 55)
    embed_dim   : dimension latente par nœud
    n_layers    : nombre de couches de message passing
    hidden_dim  : dimension interne du MLP de mise à jour
    layerNorm   : utiliser LayerNorm après chaque couche (conseillé)

    Usage identique à ResNet :
        model = LearnedGraphNet(input_dim=55, embed_dim=32,
                                n_layers=4, hidden_dim=64)
        y = model(x)           # x : (batch, 55) → y : (batch, 55)
    """
    def __init__(self, input_dim, embed_dim, n_layers, hidden_dim,
                 layerNorm=True):
        nn.Module.__init__(self)
        _wrapper.__init__(self)

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_layers  = n_layers
        self.hidden_dim = hidden_dim
        self.layerNorm  = layerNorm

        # Graphe appris
        self.graph = _LearnedGraph(input_dim)

        # Embedding scalaire → vecteur
        self.node_embedding = _NodeEmbedding(embed_dim)

        # Couches de message passing
        self.gnn = _GNNStack(n_layers, embed_dim, hidden_dim,
                             layerNorm=layerNorm)

        # Projection finale : (n_nodes * embed_dim) → input_dim
        self.output_proj = nn.Linear(input_dim * embed_dim, input_dim)

    # ── Interface _wrapper ────────────────────────────────────────────────
    def _get_config(self):
        return {
            "input_dim":  self.input_dim,
            "embed_dim":  self.embed_dim,
            "n_layers":   self.n_layers,
            "hidden_dim": self.hidden_dim,
            "layerNorm":  self.layerNorm,
        }

    @classmethod
    def _from_config(cls, config):
        return cls(**config)

    # ── Forward ───────────────────────────────────────────────────────────
    def torchForward(self, x):
        """
        x : (batch, input_dim)
        retourne : (batch, input_dim)
        """
        B = x.shape[0]

        # 1. Calculer la matrice d'adjacence (partagée sur tout le batch)
        A = self.graph()                          # (N, N)

        # 2. Embedding de chaque nœud scalaire
        h = self.node_embedding(x)                # (B, N, embed_dim)

        # 3. Message passing
        h = self.gnn(h, A)                        # (B, N, embed_dim)

        # 4. Lecture : aplatir et projeter vers input_dim
        h = h.reshape(B, -1)                      # (B, N * embed_dim)
        return self.output_proj(h)                # (B, input_dim)

    # ── Utilitaires ───────────────────────────────────────────────────────
    def get_learned_graph(self):
        """
        Retourne la matrice d'adjacence apprise (N, N) en numpy,
        prête à être visualisée avec imshow / seaborn.

        Exemple :
            adj = model.get_learned_graph()
            plt.imshow(adj, cmap='hot')
        """
        return self.graph.get_adjacency().cpu().numpy()

    def enableFineTuning(self, freeze_graph=False, freeze_embedding=False,
                         freeze_gnn=False, freeze_output=False):
        if freeze_graph:
            for p in self.graph.parameters():
                p.requires_grad = False
        if freeze_embedding:
            for p in self.node_embedding.parameters():
                p.requires_grad = False
        if freeze_gnn:
            for p in self.gnn.parameters():
                p.requires_grad = False
        if freeze_output:
            for p in self.output_proj.parameters():
                p.requires_grad = False

    def disableFineTuning(self):
        for p in self.parameters():
            p.requires_grad = True
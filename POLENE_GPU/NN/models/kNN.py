from POLENE_GPU.NN.models._wrapper import _wrapper

import os
import math
import pickle

import torch
import torch.nn as nn


class KNN(_wrapper):
    """
    kNN regressor 100% torch (CPU ou GPU), même interface que les modèles NN (_wrapper).

    Paramètres identiques à sklearn.neighbors.KNeighborsRegressor :
        n_neighbors, weights ('uniform' | 'distance' | callable), algorithm, leaf_size,
        p, metric, metric_params, n_jobs.

    Métriques supportées : 'minkowski' (avec p), 'euclidean'/'l2', 'manhattan'/'cityblock'/'l1',
    'chebyshev'/'infinity', 'cosine'.

    Notes sur `algorithm` / `leaf_size` / `n_jobs` :
        Ils sont acceptés pour la compatibilité d'interface, mais la recherche est toujours
        exacte et en force brute par blocs (cdist + topk) sur le device des données : sur GPU
        c'est en général plus rapide qu'un arbre (le parcours d'arbre est peu parallélisable),
        et les voisins renvoyés sont les mêmes que ceux de sklearn (exact).

    Options supplémentaires :
        max_elements : taille max (nb de distances) d'un bloc de calcul, pour borner la mémoire.
        use_mm       : si True, utilise la formulation matricielle pour la distance euclidienne
                       (plus rapide, un peu moins précise en float32).

    Mode incréments (predictState=False) :
        le modèle prédit omega_hat = outputScaler(chem(y) - chem(x)). Utiliser `fitFromStates`
        pour construire la cible depuis (x, y) physiques, et `predictStates` pour revenir à l'état.
    """

    _EUCLID = ("euclidean", "l2")
    _MANHAT = ("manhattan", "cityblock", "l1")
    _CHEBY = ("chebyshev", "infinity")

    def __init__(
            self,
            n_neighbors=20,
            weights="distance",
            algorithm="auto",
            leaf_size=30,
            p=2,
            metric="minkowski",
            metric_params=None,
            n_jobs=None,
            predictState=True,
            chemScaler=None,
            chemStdScaler=None,
            max_elements=2 ** 27,
            use_mm=False,
    ):
        nn.Module.__init__(self)
        _wrapper.__init__(self, predictState, chemScaler, chemStdScaler)

        if algorithm not in ("auto", "ball_tree", "kd_tree", "brute"):
            raise ValueError(f"algorithm invalide : {algorithm}")
        if not (weights in ("uniform", "distance") or callable(weights)):
            raise ValueError(f"weights invalide : {weights}")
        if callable(metric):
            raise NotImplementedError("metric callable non supportée")
        if metric_params:
            raise NotImplementedError("metric_params non supporté")
        if n_neighbors < 1:
            raise ValueError("n_neighbors doit être >= 1")

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.max_elements = max_elements
        self.use_mm = use_mm

        self._minkowski_p = self._resolve_p()

        # données d'entraînement : buffers (suivent .to(device), présents dans le state_dict)
        self.register_buffer("_X", None)
        self.register_buffer("_y", None)
        self.isFitted = False

    # ------------------------------------------------------------- distances
    def _resolve_p(self):
        m = self.metric
        if m == "minkowski":
            return float(self.p)
        if m in self._EUCLID:
            return 2.0
        if m in self._MANHAT:
            return 1.0
        if m in self._CHEBY:
            return math.inf
        if m == "cosine":
            return None
        raise ValueError(f"metric non supportée : {m}")

    def _distance(self, a, b):
        """Matrice de distances (len(a), len(b))."""
        if self.metric == "cosine":
            an = torch.nn.functional.normalize(a, dim=1)
            bn = torch.nn.functional.normalize(b, dim=1)
            return (1.0 - an @ bn.T).clamp_min(0.0)
        p = self._minkowski_p
        if p == 2.0:
            mode = "use_mm_for_euclid_dist" if self.use_mm else "donot_use_mm_for_euclid_dist"
            return torch.cdist(a, b, p=2.0, compute_mode=mode)
        return torch.cdist(a, b, p=p)

    # ------------------------------------------------------------------- fit
    @torch.no_grad()
    def fit(self, X, y, tensorInput=False):
        """
        Mémorise (X, y). y est la cible telle que le modèle doit la sortir
        (états scalés si predictState=True, omega_hat si predictState=False).
        X, y : objets `serie` (tensorInput=False) ou tenseurs (tensorInput=True).
        """
        if not tensorInput:
            X, y = X.data, y.data
        X = torch.as_tensor(X).detach()
        y = torch.as_tensor(y).detach()
        if y.ndim == 1:
            y = y[:, None]
        if X.shape[0] != y.shape[0]:
            raise ValueError("X et y doivent avoir le même nombre de lignes")
        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors > nombre d'échantillons d'entraînement")

        self._X = X.contiguous()
        self._y = y.to(device=X.device, dtype=X.dtype).contiguous()
        self.isFitted = True
        return self

    @torch.no_grad()
    def fitFromStates(self, X, x_state, y_state, tensorInput=False):
        """
        Fit à partir d'états physiques.
            X       : entrée du modèle (features, déjà scalées si besoin)
            x_state : état physique avant
            y_state : état physique après
        predictState=True  -> cible = y_state
        predictState=False -> cible = outputScaler( chem(y_state) - chem(x_state) )
        """
        if not tensorInput:
            X, x_state, y_state = X.data, x_state.data, y_state.data
        if self.predictState:
            return self.fit(X, y_state, tensorInput=True)
        if self.outputScaler is None:
            raise ValueError("outputScaler requis pour fitFromStates avec predictState=False")
        omega = self.torchState2Omega(x_state, y_state, dtype=X.dtype)
        target = self.outputScaler.transformTorch(omega)
        return self.fit(X, target, tensorInput=True)

    # ------------------------------------------------------------- recherche
    @torch.no_grad()
    def _kneighbors_block(self, xq, k):
        d = self._distance(xq, self._X)
        dist, idx = torch.topk(d, k, dim=1, largest=False, sorted=True)
        return dist, idx

    @torch.no_grad()
    def kneighbors(self, x, n_neighbors=None, return_distance=True):
        """Équivalent de sklearn.kneighbors. x : tenseur. Retourne (dist, idx) ou idx."""
        self._check_fitted()
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        if k > self._X.shape[0]:
            raise ValueError("n_neighbors > nombre d'échantillons d'entraînement")
        out_dev = x.device
        xq = x.to(device=self._X.device, dtype=self._X.dtype)
        bs = max(1, self.max_elements // self._X.shape[0])
        dists, idxs = [], []
        for s in range(0, xq.shape[0], bs):
            d, i = self._kneighbors_block(xq[s:s + bs], k)
            dists.append(d)
            idxs.append(i)
        dist, idx = torch.cat(dists), torch.cat(idxs)
        if return_distance:
            return dist.to(out_dev), idx.to(out_dev)
        return idx.to(out_dev)

    def _weights(self, dist):
        if self.weights == "uniform":
            return None
        if self.weights == "distance":
            zero = dist == 0
            any_zero = zero.any(dim=1, keepdim=True)
            inv = 1.0 / torch.where(zero, torch.ones_like(dist), dist)
            return torch.where(any_zero, zero.to(dist.dtype), inv)
        return self.weights(dist)

    # --------------------------------------------------------------- forward
    @torch.no_grad()
    def torchForward(self, x):
        self._check_fitted()
        xq = x.to(device=self._X.device, dtype=self._X.dtype)
        k = self.n_neighbors
        bs = max(1, self.max_elements // self._X.shape[0])
        out = []
        for s in range(0, xq.shape[0], bs):
            dist, idx = self._kneighbors_block(xq[s:s + bs], k)
            neigh = self._y[idx]                       # (b, k, d)
            w = self._weights(dist)
            if w is None:
                pred = neigh.mean(dim=1)
            else:
                pred = (neigh * w[..., None]).sum(dim=1) / w.sum(dim=1, keepdim=True)
            out.append(pred)
        return torch.cat(out).to(device=x.device, dtype=x.dtype)

    def predictStates(self, X, x_state, tensorInput=False):
        """
        Prédit directement l'état suivant (physique si predictState=False, scalé sinon).
            predictState=True  -> sortie du modèle
            predictState=False -> output2State(x_state, omega_hat)
        """
        if self.predictState:
            return self(X, tensorInput=tensorInput)
        omega_hat = self(X, tensorInput=tensorInput)
        return self.output2State(x_state, omega_hat, tensorInput=tensorInput)

    def _check_fitted(self):
        if not self.isFitted or self._X is None:
            raise RuntimeError("KNN: appeler fit(...) avant l'inférence.")

    # ------------------------------------------------------- config / saving
    def _get_config(self):
        return {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "metric": self.metric,
            "metric_params": self.metric_params,
            "n_jobs": self.n_jobs,
            "predictState": self.predictState,
            "max_elements": self.max_elements,
            "use_mm": self.use_mm,
        }

    @classmethod
    def _from_config(cls, config):
        return cls(**config)

    def save(self, dir, file):
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, file)
        checkpoint = {
            "state_dict": self.state_dict(),   # contient _X et _y une fois fitté
            "config": self._get_config(),
            "scalers": {
                "input": pickle.dumps(self.inputScaler),
                "output": pickle.dumps(self.outputScaler),
                "chem": pickle.dumps(self.chemScaler),
                "chemStd": pickle.dumps(self.chemStdScaler),
            },
            "isFitted": self.isFitted,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, weights_only=False)
        model = cls._from_config(checkpoint["config"])
        sd = dict(checkpoint["state_dict"])
        for name in ("_X", "_y"):
            if name in sd:
                setattr(model, name, sd.pop(name))
        model.load_state_dict(sd)
        sc = checkpoint["scalers"]
        model.inputScaler = pickle.loads(sc["input"])
        model.outputScaler = pickle.loads(sc["output"])
        model.chemScaler = pickle.loads(sc["chem"]) if "chem" in sc else None
        model.chemStdScaler = pickle.loads(sc["chemStd"]) if "chemStd" in sc else None
        model.isFitted = checkpoint["isFitted"]
        return model

    # ----------------------------------------- compat avec les routines NN
    def enableFineTuning(self, *args, **kwargs):
        pass

    def disableFineTuning(self):
        pass
from POLENE_GPU.NN.trainer._statsSave import _statsSave

import torch
import torch.nn as nn
from torch.func import jacrev, vmap
from tqdm import tqdm
import gc

class SobolevTrainer(_statsSave):
    """
    Sobolev Training — Czarnecki et al., 2017  (https://arxiv.org/abs/1706.04859)

    Loss:
        L(θ) = (1/N) Σ_i  ‖f(xᵢ) − m(xᵢ|θ)‖²
                         + ‖∇_x f(xᵢ) − ∇_x m(xᵢ|θ)‖²

    Les deux termes ont le même poids (pas de λ), conformément à l'article.
    Les jacobiennes sont complètes (shape: N × d_out × d_in) et calculées en X.
    """

    def __init__(
        self,
        testSize=0.25,
        lr=5e-3,
        warmupSteps=2000,
        decayRate=0.92,
        decaySteps=1500,
        epochs=50,
        batchSize=128,
        trainingDevice=None,
        trainingDtype=None,
    ):
        _statsSave.__init__(self)
        self.testSize = testSize
        self.lr = lr
        self.warmupSteps = warmupSteps
        self.decayRate = decayRate
        self.decaySteps = decaySteps
        self.epochs = epochs
        self.batchSize = batchSize
        self.trainingDevice = trainingDevice
        self.trainingDtype = trainingDtype

    # ── LR schedule (identique à JVPTrainer) ─────────────────────────────────

    def warmup_exp_decay(self, k):
        if self.warmupSteps > 0 and k < self.warmupSteps:
            return (k + 1) / self.warmupSteps
        return self.decayRate ** ((k - self.warmupSteps) / self.decaySteps)

    # ── Train / test split (identique à JVPTrainer) ──────────────────────────

    def trainTestSplit(self, X, Y, J):
        if X.nSamples != Y.nSamples or J.nSamples != X.nSamples:
            raise ValueError("X, Y and J must have the same number of samples")

        N = X.nSamples
        n_test = int(N * self.testSize)
        indices = torch.randperm(N)

        test_idx  = indices[:n_test]
        train_idx = indices[n_test:]

        return (
            X[train_idx].contiguous(), X[test_idx].contiguous(),
            Y[train_idx].contiguous(), Y[test_idx].contiguous(),
            J[train_idx].contiguous(), J[test_idx].contiguous(),
        )

    # ── Sobolev loss ─────────────────────────────────────────────────────────

    @staticmethod
    def sobolev_loss(model, batch_X, batch_Y, target_jacobians):
        """
        Calcule la loss de Sobolev exacte du papier.

        Args:
            model             : le réseau neuronal
            batch_X           : (N, d_in)  — points d'évaluation
            batch_Y           : (N, d_out) — valeurs cibles  f(xᵢ)
            target_jacobians  : (N, d_out, d_in) — jacobiennes cibles  ∇_x f(xᵢ)

        Returns:
            loss_value   : scalaire
            loss_value_term  : MSE valeurs (pour logging)
            loss_jacob_term  : MSE jacobiennes (pour logging)
        """
        # ── Terme 1 : MSE sur les valeurs ─────────────────────────────────
        preds = model(batch_X)             # (N, d_out)
        loss_values = nn.functional.mse_loss(preds.data, batch_Y.data)

        # ── Terme 2 : MSE sur les jacobiennes complètes ───────────────────
        # On a besoin du gradient par rapport à x → requires_grad sur x
        x = batch_X.data.detach().requires_grad_(True)       # (N, d_in)

        def model_fn(single_x):
            """Appliqué sample par sample via vmap."""
            return model(single_x.unsqueeze(0), tensorInput=True).squeeze(0)

        # jacrev calcule la jacobienne complète d_out × d_in pour un seul x
        # vmap vectorise sur le batch  →  (N, d_out, d_in)
        batch_jacobian_fn = vmap(jacrev(model_fn))
        pred_jacobians = batch_jacobian_fn(x)                # (N, d_out, d_in)

        loss_jacobians = nn.functional.mse_loss(pred_jacobians, target_jacobians.data)

        # ── Loss totale (poids égaux, comme dans le papier) ───────────────
        lamb = 0.5
        loss = lamb/2 * loss_values + (1-lamb)/2 * loss_jacobians

        return loss, loss_values.detach(), loss_jacobians.detach()

    # ── Training phase ────────────────────────────────────────────────────────

    def __trainingPhase(self, X, Y, target_jacobians, model, optimizer, lrScheduler):
        model.train()
        totalLoss        = 0.0
        totalLossValues  = 0.0
        totalLossJacob   = 0.0

        for batch in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            batch_X = X[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_Y = Y[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_J = target_jacobians[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            # batch_J : (B, d_out, d_in) — jacobiennes cibles ∇_x f(xᵢ)

            loss, lv, lj = self.sobolev_loss(model, batch_X, batch_Y, batch_J)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrScheduler.step()

            n = batch_X.data.size(0)
            totalLoss       += loss.item() * n
            totalLossValues += lv.item()   * n
            totalLossJacob  += lj.item()   * n

        N = X.nSamples
        totalLoss       /= N
        totalLossValues /= N
        totalLossJacob  /= N

        self.stats.loss.append(totalLoss)
        return totalLoss, totalLossValues, totalLossJacob

    # ── Validation phase ──────────────────────────────────────────────────────

    def __validationPhase(self, Xval, Yval, model, lossFunc):
        model.eval()
        totalLoss = 0.0

        with torch.no_grad():
            for batch in range(0, Xval.nSamples, self.batchSize):
                batch_X = Xval[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
                batch_Y = Yval[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

                preds = model(batch_X)
                loss  = lossFunc(preds.data, batch_Y.data)
                totalLoss += loss.item() * batch_X.data.size(0)

        totalLoss /= Xval.nSamples
        self.stats.valLoss.append(totalLoss)
        return totalLoss

    # ── Point d'entrée public ─────────────────────────────────────────────────

    def train(self, X, Y, target_jacobians, model):
        """
        Args:
            X                 : dataset d'entrées
            Y                 : dataset de valeurs cibles  f(xᵢ)
            target_jacobians  : Tensor (N, d_out, d_in) — ∇_x f(xᵢ) pour tout i
            model             : le réseau neuronal
        """
        torch.cuda.empty_cache()
        gc.collect()

        # Split — on splitte X, Y et les jacobiennes avec les mêmes indices
        X_train, X_val, Y_train, Y_val, J_train, J_val = self.trainTestSplit(X, Y, target_jacobians)

        model.to(self.trainingDevice).to(self.trainingDtype)

        optimizer    = torch.optim.Adam(model.parameters(), lr=self.lr)
        lrScheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[self.warmup_exp_decay])
        lossFunc     = nn.MSELoss()

        for epoch in range(self.epochs):
            trainLoss, lv, lj = self.__trainingPhase(
                X_train, Y_train, J_train, model, optimizer, lrScheduler
            )
            valLoss = self.__validationPhase(X_val, Y_val, model, lossFunc)

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: {self._format_loss(trainLoss)}"
                f"  (val: {self._format_loss(lv)}  |  jac: {self._format_loss(lj)})  |  "
                f"Val MSE: {self._format_loss(valLoss)}  |  "
                f"LR: {current_lr:.2e}"
            )



class JVPTrainer(_statsSave):
    def __init__(self, testSize=0.25, lr=5e-3, warmupSteps=2000, decayRate=0.92, decaySteps=1500, epochs=50, batchSize=128, trainingDevice=None, trainingDtype=None):
        _statsSave.__init__(self)
        self.testSize = testSize
        self.lr = lr
        self.warmupSteps = warmupSteps
        self.decayRate = decayRate
        self.decaySteps = decaySteps

        self.epochs = epochs
        self.batchSize = batchSize

        self.trainingDevice = trainingDevice
        self.trainingDtype = trainingDtype

    def warmup_exp_decay(self, k):
        if self.warmupSteps > 0 and k < self.warmupSteps:
            return (k + 1) / self.warmupSteps
        return self.decayRate ** ((k - self.warmupSteps) / self.decaySteps)

    def trainTestSplit(self, X, Y):
        if X.nSamples != Y.nSamples:
            raise ValueError("X and Y must have the same number of samples")

        N = X.nSamples
        n_test = int(N * self.testSize)

        indices = torch.arange(N)
        indices = indices[torch.randperm(N)]

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        X_train = X[train_idx].contiguous()
        X_test = X[test_idx].contiguous()
        Y_train = Y[train_idx].contiguous()
        Y_test = Y[test_idx].contiguous()

        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def physics_loss_jvp(model, Y, preds):
        # ── JVP du réseau dans l'espace normalisé ─────────────────────────
        def model_fn(x):
            return model(x, tensorInput=True)

        _, Jv_network = torch.func.jvp(
            model_fn,
            (Y.data,),
            (preds.data - Y.data,)
        )  # (N, d)

        return (Jv_network ** 2).mean()

    def __trainingPhase(self, X, Y, model, optimizer, lrScheduler):
        model.train()
        totalLoss = 0

        for batch in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            batch_X = X[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_Y = Y[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            # ── Loss données ───────────────────────────────────────────────
            preds = model(batch_X)

            # ── Loss physique ──────────────────────────────────────────────
            loss = self.physics_loss_jvp(model, batch_Y, preds) + nn.MSELoss()(preds.data, batch_Y.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrScheduler.step()

            totalLoss += loss.item() * batch_X.data.size(0)
            #print(loss.item(), lossData.item(), lambdaPhysics * lossPhysics.item())

        totalLoss /= X.nSamples
        self.stats.loss.append(totalLoss)
        return totalLoss


    def __validationPhase(self, Xval, Yval, model, optimizer, lrScheduler, lossFunc):
        model.eval()
        totalLoss = 0.0

        with torch.no_grad():
            for batch in range(0, Xval.nSamples, self.batchSize):
                batch_X = Xval[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
                batch_Y = Yval[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

                preds = model(batch_X)
                loss = lossFunc(preds.data, batch_Y.data)
                totalLoss += loss.item() * batch_X.data.size(0)

        totalLoss /= Xval.nSamples
        self.stats.valLoss.append(totalLoss)

        return totalLoss


    def train(self, X, Y, model):
        torch.cuda.empty_cache()
        gc.collect()

        X, Xval, Y, Yval = self.trainTestSplit(X, Y)
        #  data is already shuffled.

        model.to(self.trainingDevice).to(self.trainingDtype)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[self.warmup_exp_decay])
        lossFunc = nn.MSELoss()

        for epoch in range(self.epochs):

            trainingLoss = self.__trainingPhase(X, Y, model, optimizer, lrScheduler)
            validationLoss = self.__validationPhase(Xval, Yval, model, optimizer, lrScheduler, lossFunc)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{epoch + 1}/{self.epochs}] "
                  f"Train: {self._format_loss(trainingLoss)}  |  "
                  f"Val: {self._format_loss(validationLoss)}  |  "
                  f"LR: {current_lr:.2e}")


class JacobianGradNormTrainer(_statsSave):
    """
    Minimise deux objectifs via GradNorm :
      - L_mse  = MSE(pred, Y)
      - L_jac  = ||J_pred·δ − J_true·δ||²   avec δ = pred − Y
                 J_pred est la jacobienne autograd du réseau (via JVP)
                 J_true est la jacobienne physique fournie par le dataset

    Les poids w1, w2 sont appris via GradNorm + EMA des ratios de loss.
    """
    # This trainer does not really work whene we study the linearisation error around the network prediction error. It is not improved compared to a traditional network

    def __init__(
        self,
        testSize=0.25,
        lossWeightJac=1.0e-4,
        emaBeta=0.99,
        gradNormAlpha=0.5,
        lr=5e-3,
        warmupSteps=2000,
        decayRate=0.92,
        decaySteps=1500,
        epochs=50,
        batchSize=128,
        trainingDevice=None,
        trainingDtype=None,
    ):
        _statsSave.__init__(self)
        self.testSize = testSize
        self.lr = lr
        self.warmupSteps = warmupSteps
        self.decayRate = decayRate
        self.decaySteps = decaySteps
        self.epochs = epochs
        self.batchSize = batchSize
        self.trainingDevice = trainingDevice
        self.trainingDtype = trainingDtype

        self.lossWeightJac = lossWeightJac   # prefacteur initial sur L_jac avant normalisation GradNorm
        self.alpha = gradNormAlpha

        # Poids apprenables pour GradNorm
        self.taskWeights = torch.nn.Parameter(
            torch.ones(2, device=self.trainingDevice)
        )
        self.weightOptimizer = torch.optim.Adam([self.taskWeights], lr=1e-3)

        # EMA pour estimer le taux de progrès relatif de chaque tâche
        self.emaBeta = emaBeta
        self.emaMse = None
        self.emaJac = None

    # ─────────────────────────────────────────────────────────────── scheduler

    def warmup_exp_decay(self, k):
        if self.warmupSteps > 0 and k < self.warmupSteps:
            return (k + 1) / self.warmupSteps
        return self.decayRate ** ((k - self.warmupSteps) / self.decaySteps)

    # ──────────────────────────────────────────────────────────────── split

    def trainTestSplit(self, X, Y, J):
        if X.nSamples != Y.nSamples or X.nSamples != J.nSamples:
            raise ValueError("X, Y et J doivent avoir le même nombre d'échantillons")

        N = X.nSamples
        n_test = int(N * self.testSize)
        indices = torch.randperm(N)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        return (
            X[train_idx].contiguous(), X[test_idx].contiguous(),
            Y[train_idx].contiguous(), Y[test_idx].contiguous(),
            J[train_idx].contiguous(), J[test_idx].contiguous(),
        )

    # ──────────────────────────────────────────────────────────── loss jacobienne

    @staticmethod
    def jacobian_consistency_loss(model, preds, Y, J_true):
        """
        Calcule ||J_pred·δ − J_true·δ||²  avec δ = pred − Y.

        J_pred·δ  est obtenu par JVP (produit jacobien-vecteur sans
        construire la jacobienne entière).
        J_true·δ  est un simple bmm.

        Args:
            model   : le réseau (appelé via tensorInput=True)
            preds   : sortie du modèle  (N, d_out)  – avec .data attribut
            Y       : cibles             (N, d_out)
            J_true  : jacobienne vraie   (N, d_out, d_in)  – NB: J @ δ_in
                      Si J_true est de shape (N, d, d) (carré), on l'utilise
                      directement comme opérateur linéaire sur δ_out.
        """
        delta = Y.data - preds.data            # (N, d)

        # ── J_pred · δ  via JVP ────────────────────────────────────────────
        def model_fn(x):
            return model(x, tensorInput=True)

        _, Jv_pred = torch.func.jvp(
            model_fn,
            (Y.data,),
            (delta,),
        )                                        # (N, d_out)

        # ── J_true · δ  via bmm ────────────────────────────────────────────
        # J_true : (N, d_out, d_in) ;  delta : (N, d_in)
        delta_col = delta.unsqueeze(-1)          # (N, d, 1)
        Jv_true = torch.bmm(J_true, delta_col).squeeze(-1)  # (N, d_out)

        return ((Jv_pred - Jv_true) ** 2).mean()

    # ──────────────────────────────────────────────────────────────── GradNorm

    @staticmethod
    def _grad_norm(loss, params):
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, create_graph=True
        )
        return torch.norm(
            torch.stack([g.norm() for g in grads if g is not None])
        )

    def _update_ema(self, mse_val, jac_val):
        mse = mse_val.detach()
        jac = jac_val.detach()
        if self.emaMse is None:
            self.emaMse, self.emaJac = mse, jac
        else:
            b = self.emaBeta
            self.emaMse = b * self.emaMse + (1 - b) * mse
            self.emaJac = b * self.emaJac + (1 - b) * jac
        mean = (self.emaMse + self.emaJac) / 2.0 + 1e-12
        return self.emaMse / mean, self.emaJac / mean

    # ──────────────────────────────────────────────────────── training phase

    def __trainingPhase(self, X, Y, J_Y, model, optimizer, lrScheduler):
        model.train()
        totalLoss = 0.0
        totalMse = 0.0
        totalJac = 0.0

        for batch in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            batch_X   = X  [batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_Y   = Y  [batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_J_Y = J_Y[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            preds = model(batch_X)

            # ── losses individuelles ────────────────────────────────────────
            mseLoss = nn.MSELoss()(preds.data, batch_Y.data)
            jacLoss = self.lossWeightJac * self.jacobian_consistency_loss(
                model, preds, batch_Y, batch_J_Y.data
            )

            # ── poids softmax ───────────────────────────────────────────────
            weights = torch.softmax(self.taskWeights, dim=0)
            weighted_mse = weights[0] * mseLoss
            weighted_jac = weights[1] * jacLoss
            loss = weighted_mse + weighted_jac

            # ── GradNorm ────────────────────────────────────────────────────
            shared_params = [p for p in model.parameters() if p.requires_grad]
            G1 = self._grad_norm(weighted_mse, shared_params)
            G2 = self._grad_norm(weighted_jac, shared_params)
            G_avg = (G1 + G2) / 2.0

            r1, r2 = self._update_ema(mseLoss, jacLoss)

            grad_loss = (
                torch.abs(G1 - G_avg * (r1 ** self.alpha)) +
                torch.abs(G2 - G_avg * (r2 ** self.alpha))
            )

            self.weightOptimizer.zero_grad()
            grad_loss.backward(retain_graph=True)
            self.weightOptimizer.step()

            # ── mise à jour du modèle ───────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrScheduler.step()

            bs = batch_X.data.size(0)
            totalLoss += loss.item()  * bs
            totalMse  += mseLoss.item() * bs
            totalJac  += jacLoss.item() * bs

        N = X.nSamples
        totalLoss /= N;  totalMse /= N;  totalJac /= N
        self.stats.loss.append(totalLoss)
        return totalLoss, totalMse, totalJac

    # ─────────────────────────────────────────────────────── validation phase

    def __validationPhase(self, Xval, Yval, model, lossFunc):
        model.eval()
        totalLoss = 0.0

        with torch.no_grad():
            for batch in range(0, Xval.nSamples, self.batchSize):
                batch_X = Xval[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
                batch_Y = Yval[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

                preds = model(batch_X)
                loss = lossFunc(preds.data, batch_Y.data)
                totalLoss += loss.item() * batch_X.data.size(0)

        totalLoss /= Xval.nSamples
        self.stats.valLoss.append(totalLoss)
        return totalLoss

    # ──────────────────────────────────────────────────────────────── train

    def train(self, X, Y, J_Y, model):
        torch.cuda.empty_cache()
        gc.collect()

        X, Xval, Y, Yval, J_Y, J_Y_val = self.trainTestSplit(X, Y, J_Y)

        model.to(self.trainingDevice).to(self.trainingDtype)

        optimizer    = torch.optim.Adam(model.parameters(), lr=self.lr)
        lrScheduler  = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[self.warmup_exp_decay]
        )
        lossFunc = nn.MSELoss()

        for epoch in range(self.epochs):
            trainLoss, mseLoss, jacLoss = self.__trainingPhase(
                X, Y, J_Y, model, optimizer, lrScheduler
            )
            valLoss = self.__validationPhase(Xval, Yval, model, lossFunc)

            w = torch.softmax(self.taskWeights, dim=0).detach()
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: {self._format_loss(trainLoss)}  "
                f"(mse={self._format_loss(mseLoss)}, jac={self._format_loss(jacLoss)})  |  "
                f"Val: {self._format_loss(valLoss)}  |  "
                f"LR: {current_lr:.2e}  |  "
                f"w_mse={w[0]:.3f}  w_jac={w[1]:.3f}"
            )

class dataJVPTrainer(_statsSave):
    def __init__(
        self,
        testSize=0.25,
        lr=5e-3,
        warmupSteps=2000,
        decayRate=0.92,
        decaySteps=1500,
        epochs=50,
        batchSize=128,
        trainingDevice=None,
        trainingDtype=None,
    ):
        _statsSave.__init__(self)
        self.testSize = testSize
        self.lr = lr
        self.warmupSteps = warmupSteps
        self.decayRate = decayRate
        self.decaySteps = decaySteps
        self.epochs = epochs
        self.batchSize = batchSize
        self.trainingDevice = trainingDevice
        self.trainingDtype = trainingDtype

    # ─────────────────────────────────────────────────────────────── scheduler

    def warmup_exp_decay(self, k):
        if self.warmupSteps > 0 and k < self.warmupSteps:
            return (k + 1) / self.warmupSteps
        return self.decayRate ** ((k - self.warmupSteps) / self.decaySteps)

    # ──────────────────────────────────────────────────────────────── split

    def trainTestSplit(self, X, Y, J):
        if X.nSamples != Y.nSamples or X.nSamples != J.nSamples:
            raise ValueError("X, Y et J doivent avoir le même nombre d'échantillons")

        N = X.nSamples
        n_test = int(N * self.testSize)
        indices = torch.randperm(N)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        return (
            X[train_idx].contiguous(), X[test_idx].contiguous(),
            Y[train_idx].contiguous(), Y[test_idx].contiguous(),
            J[train_idx].contiguous(), J[test_idx].contiguous(),
        )

    # ──────────────────────────────────────────────────────────── loss jacobienne

    @staticmethod
    def physics_loss_jvp(model, X, Y, J_true):

        DeltaX = (Y.data - X.data)

        # ── JVP du réseau dans l'espace normalisé ─────────────────────────
        def model_fn(x):
            return model(x, tensorInput=True)

        _, Jv_network = torch.func.jvp(
            model_fn,
            (X.data,),
            (DeltaX,)
        )  # (N, d)

        Jv_true = torch.bmm(J_true.data, DeltaX.unsqueeze(-1))

        return nn.MSELoss()(Jv_network, Jv_true.squeeze(-1))

    # ──────────────────────────────────────────────────────── training phase

    def __trainingPhase(self, X, Y, J_Y, model, optimizer, lrScheduler):
        model.train()
        totalLoss = 0.0

        for batch in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            batch_X   = X[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_Y   = Y[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_J_Y = J_Y[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            loss = self.physics_loss_jvp(model, batch_X, batch_Y, batch_J_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrScheduler.step()

            totalLoss += loss.item() * batch_X.data.size(0)

        totalLoss /= X.nSamples
        self.stats.loss.append(totalLoss)
        return totalLoss

    # ─────────────────────────────────────────────────────── validation phase

    def __validationPhase(self, Xval, Yval, model, lossFunc):
        model.eval()
        totalLoss = 0.0

        with torch.no_grad():
            for batch in range(0, Xval.nSamples, self.batchSize):
                batch_X = Xval[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
                batch_Y = Yval[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

                preds = model(batch_X)
                loss = lossFunc(preds.data, batch_Y.data)
                totalLoss += loss.item() * batch_X.data.size(0)

        totalLoss /= Xval.nSamples
        self.stats.valLoss.append(totalLoss)
        return totalLoss

    # ──────────────────────────────────────────────────────────────── train

    def train(self, X, Y, J_Y, model):
        torch.cuda.empty_cache()
        gc.collect()

        X, Xval, Y, Yval, J_Y, J_Y_val = self.trainTestSplit(X, Y, J_Y)

        model.to(self.trainingDevice).to(self.trainingDtype)

        optimizer   = torch.optim.Adam(model.parameters(), lr=self.lr)
        lrScheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[self.warmup_exp_decay]
        )
        lossFunc = nn.MSELoss()

        for epoch in range(self.epochs):
            trainLoss = self.__trainingPhase(X, Y, J_Y, model, optimizer, lrScheduler)
            valLoss   = self.__validationPhase(Xval, Yval, model, lossFunc)

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train (jac mse): {self._format_loss(trainLoss)}  |  "
                f"Val (pred mse): {self._format_loss(valLoss)}  |  "
                f"LR: {current_lr:.2e}"
            )


# TODO: faire un trainer qui minimise ||J_pred(X-Y) - J_true(X-Y)||^2 afin d'entrainer la jacobienne dans le sens de la transformation. c'est proche des equations de
#  l'integrateur implicite et ça permet d'apprendre la jacobienne de la tranfo, ce qui est assez important pour la généralisation.
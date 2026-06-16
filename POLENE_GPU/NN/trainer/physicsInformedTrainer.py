from POLENE_GPU.NN.trainer._statsSave import _statsSave
from POLENE_GPU.NN.trainer._mixins import SchedulerMixin, SplitMixin, ValidationMixin, GradNormMixin, PCGradMixin, EffectiveRankMixin
from POLENE_GPU.NN.trainer._losses import (
    directional_sobolev_losses,
    full_sobolev_loss,
    jac_inv_loss,
    directional_sobolev_losses_cone,
    manifoldJacobianLoss,
    manifoldJacobianLoss_cone
)

import torch
import torch.nn as nn
from tqdm import tqdm
import gc


# ─────────────────────────────────────────────────────────────────────────────
class DirectionalSobolevTrainer(SchedulerMixin, SplitMixin, ValidationMixin, _statsSave, EffectiveRankMixin):
    """
    Directional Sobolev training with fixed λ weighting.

    loss = (1 - λ) * MSE(values) + λ * MSE(J·v)   with v = Y - X
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
            lmbda=0.5,
    ):
        _statsSave.__init__(self)
        if not 0.0 <= lmbda <= 1.0:
            raise ValueError("lmbda must be in [0, 1]")
        self.testSize = testSize
        self.lr = lr
        self.warmupSteps = warmupSteps
        self.decayRate = decayRate
        self.decaySteps = decaySteps
        self.epochs = epochs
        self.batchSize = batchSize
        self.trainingDevice = trainingDevice
        self.trainingDtype = trainingDtype
        self.lmbda = lmbda

    def __trainingPhase(self, X, Y, target_jacobians, mask, model, optimizer, lrScheduler):
        model.train()
        totalLoss = totalLV = totalLJ = 0.0

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bJ = target_jacobians[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bM = mask[start:start + self.batchSize].to(self.trainingDevice).to(torch.bool)

            lv, lj = directional_sobolev_losses(model, bX, bY, bJ, bM)
            loss = (1.0 - self.lmbda) * lv + self.lmbda * lj

            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lrScheduler.step()

            n = bX.data.size(0)
            totalLoss += loss.item() * n
            totalLV += lv.item() * n
            totalLJ += lj.item() * n

        N = X.nSamples
        totalLoss /= N;
        totalLV /= N;
        totalLJ /= N
        self.stats.loss.append(totalLoss)
        return totalLoss, totalLV, totalLJ

    def train(self, X, Y, J_X, model, jacobian_mask=None):
        torch.cuda.empty_cache()
        gc.collect()

        if jacobian_mask is None:
            mask_train = None
            mask_val = None
            X_train, X_val, Y_train, Y_val, J_train, J_val = self.trainTestSplit(X, Y, J_X)
        else:
            X_train, X_val, Y_train, Y_val, J_train, J_val, mask_train, mask_val = self.trainTestSplit(X, Y, J_X, jacobian_mask)
            mask_train = mask_train.data.flatten()
            mask_val = mask_val.data.flatten()

        model.to(self.trainingDevice).to(self.trainingDtype)
        optimizer, lrScheduler = self._build_optimizer(model)

        self.stats.ranks.append(self.compute_effective_ranks(model))

        for epoch in range(self.epochs):
            trainLoss, lv, lj = self.__trainingPhase(X_train, Y_train, J_train, mask_train, model, optimizer, lrScheduler)
            valLoss, jacobianValLoss = self._validation_phase_with_jacobian(X_val, Y_val, J_val, model)
            self.stats.ranks.append(self.compute_effective_ranks(model))

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: {self._format_loss(trainLoss)}"
                f"  (mse: {self._format_loss(lv)}  |  jac: {self._format_loss(lj)})  |  "
                f"Val: (mse: {self._format_loss(valLoss)}, jacobian: {self._format_loss(jacobianValLoss)})  |  "
                f"λ_val: {1 - self.lmbda:.3f}  λ_jac: {self.lmbda:.3f}  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )



# ─────────────────────────────────────────────────────────────────────────────
class DirectionalPCgradSobolevTrainer(SchedulerMixin, SplitMixin, ValidationMixin, _statsSave, EffectiveRankMixin):
    """
    Directional Sobolev training with fixed λ weighting.

    loss = (1 - λ) * MSE(values) + λ * MSE(J·v)   with v = Y - X
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
            lmbda=0.5,
            alpha=None
    ):
        _statsSave.__init__(self)
        if not 0.0 <= lmbda <= 1.0:
            raise ValueError("lmbda must be in [0, 1]")
        self.testSize = testSize
        self.lr = lr
        self.warmupSteps = warmupSteps
        self.decayRate = decayRate
        self.decaySteps = decaySteps
        self.epochs = epochs
        self.batchSize = batchSize
        self.trainingDevice = trainingDevice
        self.trainingDtype = trainingDtype
        self.lmbda = lmbda
        self.alpha = alpha

    def __trainingPhase(self, X, Y, target_jacobians, mask, model, pc_optimizer, lrScheduler):
        model.train()
        totalLoss = totalLV = totalLJ = 0.0

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(pc_optimizer.param_groups[0]['lr'])

            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bJ = target_jacobians[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bM = mask[start:start + self.batchSize].to(self.trainingDevice).to(torch.bool)

            if self.alpha is None:
                lv, lj = directional_sobolev_losses(model, bX, bY, bJ, bM)
            else:
                lv, lj = directional_sobolev_losses_cone(model, bX, bY, bJ, alpha=self.alpha)

            weighted_lv = (1.0 - self.lmbda) * lv
            weighted_lj = self.lmbda * lj

            pc_optimizer.pc_backward([weighted_lv, weighted_lj])

            #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            pc_optimizer.step()
            lrScheduler.step()

            n = bX.data.size(0)
            loss_val = weighted_lv.item() + weighted_lj.item()
            totalLoss += loss_val * n
            totalLV += lv.item() * n
            totalLJ += lj.item() * n

        N = X.nSamples
        totalLoss /= N;
        totalLV /= N;
        totalLJ /= N
        self.stats.loss.append(totalLoss)
        return totalLoss, totalLV, totalLJ

    def train(self, X, Y, J_X, model, jacobian_mask=None):
        torch.cuda.empty_cache()
        gc.collect()


        if jacobian_mask is None:
            mask_train = None
            mask_val = None
            X_train, X_val, Y_train, Y_val, J_train, J_val = self.trainTestSplit(X, Y, J_X)
        else:
            X_train, X_val, Y_train, Y_val, J_train, J_val, mask_train, mask_val = self.trainTestSplit(X, Y, J_X, jacobian_mask)
            mask_train = mask_train.data.flatten()
            mask_val = mask_val.data.flatten()

        model.to(self.trainingDevice).to(self.trainingDtype)
        base_optimizer, lrScheduler = self._build_optimizer(model)
        pc_optimizer = PCGradMixin(base_optimizer)

        self.stats.ranks.append(self.compute_effective_ranks(model))

        for epoch in range(self.epochs):
            trainLoss, lv, lj = self.__trainingPhase(X_train, Y_train, J_train, mask_train, model, pc_optimizer, lrScheduler)
            valLoss, jacobianValLoss = self._validation_phase_with_jacobian(X_val, Y_val, J_val, model)
            self.stats.ranks.append(self.compute_effective_ranks(model))

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: {self._format_loss(trainLoss)}"
                f"  (mse: {self._format_loss(lv)}  |  jac: {self._format_loss(lj)})  |  "
                f"Val: (mse: {self._format_loss(valLoss)}, jacobian: {self._format_loss(jacobianValLoss)})  |  "
                f"λ_val: {1 - self.lmbda:.3f}  λ_jac: {self.lmbda:.3f}  |  "
                f"LR: {pc_optimizer.param_groups[0]['lr']:.2e}"
            )




# ─────────────────────────────────────────────────────────────────────────────
class DirectionalGradNormSobolevTrainer(SchedulerMixin, SplitMixin, ValidationMixin, GradNormMixin, _statsSave):
    """
    Directional Sobolev training with GradNorm automatic weight balancing.
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
            alpha=0.5,
            gradnormLr=1e-3,
            emaBeta=0.99,
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
        self._gn_alpha_init = alpha
        self._gn_lr = gradnormLr
        self._gn_beta = emaBeta

    def __trainingPhase(self, X, Y, target_jacobians, model, optimizer, lrScheduler):
        model.train()
        totalLoss = totalLV = totalLJ = 0.0

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bJ = target_jacobians[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            lv, lj = directional_sobolev_losses(model, bX, bY, bJ)

            weights = self._softmax_weights()
            wLV = weights[0] * lv
            wLJ = weights[1] * lj

            loss = self._gradnorm_step(optimizer, [wLV, wLJ], [lv, lj], model, clip_grad=1.0)
            lrScheduler.step()

            n = bX.data.size(0)
            totalLoss += loss.item() * n
            totalLV += lv.item() * n
            totalLJ += lj.item() * n

        N = X.nSamples
        totalLoss /= N;
        totalLV /= N;
        totalLJ /= N
        self.stats.loss.append(totalLoss)
        return totalLoss, totalLV, totalLJ

    def train(self, X, Y, J_X, model):
        torch.cuda.empty_cache()
        gc.collect()

        X_train, X_val, Y_train, Y_val, J_train, J_val = self.trainTestSplit(X, Y, J_X)

        model.to(self.trainingDevice).to(self.trainingDtype)
        self._init_gradnorm(
            n_tasks=2,
            device=self.trainingDevice,
            dtype=self.trainingDtype,
            lr=self._gn_lr,
            alpha=self._gn_alpha_init,
            ema_beta=self._gn_beta,
        )
        optimizer, lrScheduler = self._build_optimizer(model)

        for epoch in range(self.epochs):
            trainLoss, lv, lj = self.__trainingPhase(
                X_train, Y_train, J_train, model, optimizer, lrScheduler
            )
            valLoss = self._validation_phase(X_val, Y_val, model)
            weights = self._softmax_weights().detach()

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: {self._format_loss(trainLoss)}"
                f"  (mse: {self._format_loss(lv)}  |  jac: {self._format_loss(lj)})  |  "
                f"Val MSE: {self._format_loss(valLoss)}  |  "
                f"w_val: {weights[0]:.3f}  w_jac: {weights[1]:.3f}  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )


# ─────────────────────────────────────────────────────────────────────────────
class SobolevTrainer(SchedulerMixin, SplitMixin, ValidationMixin, _statsSave):
    """
    Full Sobolev training — Czarnecki et al., 2017.

    loss = λ * MSE(values) + (1-λ) * MSE(J·v)

    Jacobians are full (N, d_out, d_in) and pre-computed at X.
    The direction v = Y - X is used for the directional JVP.
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
            lamb=1.0,
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
        self.lamb = lamb

    def __trainingPhase(self, X, Y, target_jacobians, jacobianScaler, model, optimizer, lrScheduler):
        model.train()
        totalLoss = totalLV = totalLJ = 0.0

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bJ = target_jacobians[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            loss, lv, lj = full_sobolev_loss(model, bX, bY, bJ, lamb=self.lamb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lrScheduler.step()

            n = bX.data.size(0)
            totalLoss += loss.item() * n
            totalLV += lv.item() * n
            totalLJ += lj.item() * n

        N = X.nSamples
        totalLoss /= N;
        totalLV /= N;
        totalLJ /= N
        self.stats.loss.append(totalLoss)
        return totalLoss, totalLV, totalLJ

    def train(self, X, Y, J_X, model, jacobianScaler=None):
        torch.cuda.empty_cache()
        gc.collect()

        X_train, X_val, Y_train, Y_val, J_train, J_val = self.trainTestSplit(X, Y, J_X)

        model.to(self.trainingDevice).to(self.trainingDtype)
        optimizer, lrScheduler = self._build_optimizer(model)

        for epoch in range(self.epochs):
            trainLoss, lv, lj = self.__trainingPhase(
                X_train, Y_train, J_train, jacobianScaler, model, optimizer, lrScheduler
            )
            valLoss = self._validation_phase(X_val, Y_val, model)

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: {self._format_loss(trainLoss)}"
                f"  (val: {self._format_loss(lv)}  |  jac: {self._format_loss(lj)})  |  "
                f"Val MSE: {self._format_loss(valLoss)}  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )


# ─────────────────────────────────────────────────────────────────────────────
class integratorSurogateTrainer(SchedulerMixin, SplitMixin, ValidationMixin, _statsSave):
    """
    Integrator surrogate trainer.

    Combines MSE on function values with a Jacobian inverse consistency loss:
        loss = MSE(pred, Y) + jac_inv_loss(J_pred, J_1) / scale

    The jac_inv_loss enforces J_model ≈ J_true^{-1}.
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
            jacLossScale=430,
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
        self.jacLossScale = jacLossScale

    def __trainingPhase(self, X, Y, target_jacobians, model, optimizer, lrScheduler):
        model.train()
        totalLoss = totalLJ = totalLV = 0.0

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bJ = target_jacobians[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            lj = jac_inv_loss(model, bX, bJ) / self.jacLossScale

            preds = model(bX)
            lv = nn.functional.mse_loss(bY.data, preds.data)

            loss = lj + lv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrScheduler.step()

            n = bX.data.size(0)
            totalLoss += loss.item() * n
            totalLJ += lj.item() * n
            totalLV += lv.item() * n

        N = X.nSamples
        totalLoss /= N;
        totalLJ /= N;
        totalLV /= N
        self.stats.loss.append(totalLoss)
        return totalLoss, totalLJ, totalLV

    def train(self, X, Y, J_1, model):
        torch.cuda.empty_cache()
        gc.collect()

        X_train, X_val, Y_train, Y_val, J_train, J_val = self.trainTestSplit(X, Y, J_1)

        model.to(self.trainingDevice).to(self.trainingDtype)
        optimizer, lrScheduler = self._build_optimizer(model)

        for epoch in range(self.epochs):
            trainLoss, lj, lv = self.__trainingPhase(
                X_train, Y_train, J_train, model, optimizer, lrScheduler
            )
            valLoss = self._validation_phase(X_val, Y_val, model)

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: {self._format_loss(trainLoss)}"
                f"  (mse: {self._format_loss(lv)}  |  jac: {self._format_loss(lj)})  |  "
                f"Val MSE: {self._format_loss(valLoss)}  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )


from POLENE_GPU.NN.trainer._statsSave import _statsSave
from POLENE_GPU.NN.trainer._mixins import SchedulerMixin, SplitMixin, ValidationMixin, GradNormMixin
from POLENE_GPU.NN.trainer._losses import jvp_physics_loss

import torch
import torch.nn as nn
from tqdm import tqdm
import gc


# ─────────────────────────────────────────────────────────────────────────────
class JacobianTrainer(SchedulerMixin, SplitMixin, ValidationMixin, _statsSave, EffectiveRankMixin):

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
            alpha=None
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
        self.alpha = alpha

    def __trainingPhase(self, X, Y, target_jacobians, model, optimizer, lrScheduler):
        model.train()
        totalLoss = 0.0

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bJ = target_jacobians[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            if self.alpha is None:
                loss = manifoldJacobianLoss(model, bX, bY, bJ)
            else:
                loss = manifoldJacobianLoss_cone(model, bX, bY, bJ, self.alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrScheduler.step()

            n = bX.data.size(0)
            totalLoss += loss.item() * n


        N = X.nSamples
        totalLoss /= N
        self.stats.loss.append(totalLoss)
        return totalLoss

    def train(self, X, Y, J_X, model):
        torch.cuda.empty_cache()
        gc.collect()

        X_train, X_val, Y_train, Y_val, J_train, J_val = self.trainTestSplit(X, Y, J_X)

        model.to(self.trainingDevice).to(self.trainingDtype)
        optimizer, lrScheduler = self._build_optimizer(model)

        self.stats.ranks.append(self.compute_effective_ranks(model))

        for epoch in range(self.epochs):
            trainLoss = self.__trainingPhase(
                X_train, Y_train, J_train, model, optimizer, lrScheduler
            )
            valLoss, valLossJac = self._validation_phase_with_jacobian(X_val, Y_val, J_val, model)
            self.stats.ranks.append(self.compute_effective_ranks(model))

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: {self._format_loss(trainLoss)}  |  "
                f"Val: (mse: {self._format_loss(valLoss)}, jacobian: {self._format_loss(valLossJac)})  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )


# ─────────────────────────────────────────────────────────────────────────────
class dataJVPTrainer(SchedulerMixin, SplitMixin, ValidationMixin, _statsSave):
    """
    Trains purely on the directional Jacobian loss (no MSE on values):

        loss = MSE( J_network(X)·(Y-X),  J_true·(Y-X) )

    Uses pre-computed Jacobians J_true at X.
    Validation is still evaluated with MSE on predictions.
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

    def __trainingPhase(self, X, Y, J_Y, model, optimizer, lrScheduler):
        model.train()
        totalLoss = 0.0

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX  = X  [start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY  = Y  [start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bJY = J_Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            loss = jvp_physics_loss(model, bX, bY, bJY)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrScheduler.step()

            totalLoss += loss.item() * bX.data.size(0)

        totalLoss /= X.nSamples
        self.stats.loss.append(totalLoss)
        return totalLoss

    def train(self, X, Y, J_Y, model):
        torch.cuda.empty_cache()
        gc.collect()

        X_train, X_val, Y_train, Y_val, J_train, J_val = self.trainTestSplit(X, Y, J_Y)

        model.to(self.trainingDevice).to(self.trainingDtype)
        optimizer, lrScheduler = self._build_optimizer(model)

        for epoch in range(self.epochs):
            trainLoss = self.__trainingPhase(X_train, Y_train, J_train, model, optimizer, lrScheduler)
            valLoss   = self._validation_phase(X_val, Y_val, model)

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train (jac mse): {self._format_loss(trainLoss)}  |  "
                f"Val (pred mse): {self._format_loss(valLoss)}  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )


# ─────────────────────────────────────────────────────────────────────────────
class JacobianGradNormTrainer(SchedulerMixin, SplitMixin, ValidationMixin, GradNormMixin, _statsSave):
    """
    MSE + directional Jacobian consistency loss, balanced via GradNorm.

    Jacobian loss:
        ||J_network(Y)·δ  −  J_true·δ||^2   with δ = Y - pred

    J_network·δ is obtained via JVP without building the full Jacobian.

    Note: this trainer showed limited improvement over plain MSE when the
    linearisation error is studied around the network prediction error.
    Kept for research purposes.
    """

    def __init__(
        self,
        testSize=0.25,
        lossWeightJac=1.0e-4,
        emaBeta=0.99,
        gradNormAlpha=0.5,
        gradNormLr=1e-3,
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
        self.lossWeightJac = lossWeightJac
        self._gn_alpha_init = gradNormAlpha
        self._gn_lr = gradNormLr
        self._gn_beta = emaBeta

    @staticmethod
    def _jacobian_consistency_loss(
        model,
        preds: torch.Tensor,
        Y: torch.Tensor,
        J_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        ||J_network(Y)·δ  −  J_true·δ||^2   with δ = Y - pred.
        J_network·δ via JVP, J_true·δ via bmm.
        """
        delta = Y.data - preds.data            # (B, d)

        def model_fn(x):
            return model(x, tensorInput=True)

        _, Jv_pred = torch.func.jvp(model_fn, (Y.data,), (delta,))
        Jv_true = torch.bmm(J_true, delta.unsqueeze(-1)).squeeze(-1)
        return ((Jv_pred - Jv_true) ** 2).mean()

    def __trainingPhase(self, X, Y, J_Y, model, optimizer, lrScheduler):
        model.train()
        totalLoss = totalMSE = totalJac = 0.0

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX  = X  [start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY  = Y  [start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bJY = J_Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            preds = model(bX)
            mseLoss = nn.functional.mse_loss(preds.data, bY.data)
            jacLoss = self.lossWeightJac * self._jacobian_consistency_loss(model, preds, bY, bJY.data)

            weights = self._softmax_weights()
            wMSE = weights[0] * mseLoss
            wJac = weights[1] * jacLoss

            loss = self._gradnorm_step(optimizer, [wMSE, wJac], [mseLoss, jacLoss], model)
            lrScheduler.step()

            n = bX.data.size(0)
            totalLoss += loss.item()    * n
            totalMSE  += mseLoss.item() * n
            totalJac  += jacLoss.item() * n

        N = X.nSamples
        totalLoss /= N; totalMSE /= N; totalJac /= N
        self.stats.loss.append(totalLoss)
        return totalLoss, totalMSE, totalJac

    def train(self, X, Y, J_Y, model):
        torch.cuda.empty_cache()
        gc.collect()

        X_train, X_val, Y_train, Y_val, J_train, J_val = self.trainTestSplit(X, Y, J_Y)

        model.to(self.trainingDevice).to(self.trainingDtype)
        self._init_gradnorm(
            n_tasks=2,
            device=self.trainingDevice,
            dtype=self.trainingDtype,
            lr=self._gn_lr,
            alpha=self._gn_alpha_init,
            ema_beta=self._gn_beta,
        )
        optimizer, lrScheduler = self._build_optimizer(model)

        for epoch in range(self.epochs):
            trainLoss, mseLoss, jacLoss = self.__trainingPhase(
                X_train, Y_train, J_train, model, optimizer, lrScheduler
            )
            valLoss = self._validation_phase(X_val, Y_val, model)
            weights = self._softmax_weights().detach()

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: {self._format_loss(trainLoss)}  "
                f"(mse={self._format_loss(mseLoss)}, jac={self._format_loss(jacLoss)})  |  "
                f"Val: {self._format_loss(valLoss)}  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}  |  "
                f"w_mse={weights[0]:.3f}  w_jac={weights[1]:.3f}"
            )
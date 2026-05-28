from POLENE_GPU.NN.trainer._statsSave import _statsSave
from POLENE_GPU.NN.trainer._mixins import SchedulerMixin, SplitMixin, ValidationMixin, GradNormMixin
from POLENE_GPU.NN.trainer._losses import mahalanobis_loss, blended_mahalanobis_loss, directional_sobolev_losses

import torch
import torch.nn as nn
from tqdm import tqdm
import gc


# ─────────────────────────────────────────────────────────────────────────────
class mahanalobisTrainer(SchedulerMixin, SplitMixin, ValidationMixin, _statsSave):
    """
    Pure Mahalanobis loss with M = (1-λ)I + λJ.

    Allows interpolation between MSE (λ=0) and full Mahalanobis (λ=1).
    The prefactor 1.08e-7 from the GradNorm trainer ensures the pure
    Mahalanobis loss learns at the same rate as MSE with no prefactor.

    Args:
        lamb : exponent for λ = 10^(-lamb). E.g. lamb=5 → λ=1e-5.
    """

    def __init__(
        self,
        testSize=0.25,
        lamb=5,
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
        self.lamb = 10 ** (-lamb)

    def __trainingPhase(self, X, Y, J_Y, model, optimizer, lrScheduler):
        model.train()
        totalLoss = 0.0

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX  = X  [start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY  = Y  [start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bJ  = J_Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            preds = model(bX)
            loss = 1e-3 * blended_mahalanobis_loss(
                preds.data, bY.data, bJ.data,
                self.lamb, self.trainingDevice, self.trainingDtype,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrScheduler.step()

            totalLoss += loss.item() * bX.data.size(0)

        totalLoss /= X.nSamples
        self.stats.loss.append(totalLoss)
        return totalLoss

    def train(self, X, Y, JSystem_Y, model):
        torch.cuda.empty_cache()
        gc.collect()

        X_train, X_val, Y_train, Y_val, J_train, J_val = self.trainTestSplit(X, Y, JSystem_Y)

        model.to(self.trainingDevice).to(self.trainingDtype)
        optimizer, lrScheduler = self._build_optimizer(model)

        for epoch in range(self.epochs):
            trainLoss = self.__trainingPhase(X_train, Y_train, J_train, model, optimizer, lrScheduler)
            valLoss   = self._validation_phase(X_val, Y_val, model)

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: {self._format_loss(trainLoss)}  |  "
                f"Val: {self._format_loss(valLoss)}  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )


# ─────────────────────────────────────────────────────────────────────────────
class jacobianMahanalobisTrainer(SchedulerMixin, SplitMixin, ValidationMixin, _statsSave):
    """
    Fixed-weight combination of MSE and Mahalanobis losses.
    Requires pre-computed Jacobians.

    loss = MSE + lossWeight * Mahalanobis(J)
    """

    def __init__(
        self,
        testSize=0.25,
        lossWeight=1.08e-7,
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
        self.lossWeight = lossWeight

    def __trainingPhase(self, X, Y, J_Y, model, optimizer, lrScheduler):
        model.train()
        totalLoss = totalMSE = totalMaha = 0.0

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX = X  [start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y  [start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bJ = J_Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            preds = model(bX)
            mseLoss  = nn.functional.mse_loss(preds.data, bY.data)
            mahaLoss = self.lossWeight * mahalanobis_loss(preds.data, bY.data, bJ.data)
            loss = mseLoss + mahaLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrScheduler.step()

            n = bX.data.size(0)
            totalLoss += loss.item()     * n
            totalMSE  += mseLoss.item()  * n
            totalMaha += mahaLoss.item() * n

        N = X.nSamples
        totalLoss /= N; totalMSE /= N; totalMaha /= N
        self.stats.loss.append(totalLoss)
        return totalLoss, totalMSE, totalMaha

    def train(self, X, Y, JSystem_Y, model):
        X_train, X_val, Y_train, Y_val, J_train, J_val = self.trainTestSplit(X, Y, JSystem_Y)

        model.to(self.trainingDevice).to(self.trainingDtype)
        optimizer, lrScheduler = self._build_optimizer(model)

        for epoch in range(self.epochs):
            trainLoss, mseLoss, mahaLoss = self.__trainingPhase(
                X_train, Y_train, J_train, model, optimizer, lrScheduler
            )
            valLoss = self._validation_phase(X_val, Y_val, model)

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: total={self._format_loss(trainLoss)}, "
                f"mse={self._format_loss(mseLoss)}, "
                f"maha={self._format_loss(mahaLoss)}  |  "
                f"Val: {self._format_loss(valLoss)}  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )


# ─────────────────────────────────────────────────────────────────────────────
class mahanalobisTrainerGradNorm(SchedulerMixin, SplitMixin, ValidationMixin, GradNormMixin, _statsSave):
    """
    MSE + Mahalanobis with automatic weight balancing via GradNorm.

    Weights w_mse, w_maha are learnable and updated at each batch so that
    the gradient norms of both tasks are equalised (up to the training-rate
    ratio raised to alpha).
    """

    def __init__(
        self,
        testSize=0.25,
        lossWeight=2.5e-7,
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
        self.lossWeight = lossWeight
        self._gn_beta = emaBeta
        self._gn_lr = gradNormLr
        self._gn_alpha_init = gradNormAlpha

    def __trainingPhase(self, X, Y, J_Y, model, optimizer, lrScheduler):
        model.train()
        totalLoss = totalMSE = totalMaha = 0.0

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX = X  [start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y  [start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bJ = J_Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            preds = model(bX)
            mseLoss  = nn.functional.mse_loss(preds.data, bY.data)
            mahaLoss = self.lossWeight * mahalanobis_loss(preds.data, bY.data, bJ.data)

            weights = self._softmax_weights()
            wMSE = weights[0] * mseLoss
            wMah = weights[1] * mahaLoss

            loss = self._gradnorm_step(optimizer, [wMSE, wMah], [mseLoss, mahaLoss], model)
            lrScheduler.step()

            n = bX.data.size(0)
            totalLoss += loss.item()     * n
            totalMSE  += mseLoss.item()  * n
            totalMaha += mahaLoss.item() * n

        N = X.nSamples
        totalLoss /= N; totalMSE /= N; totalMaha /= N
        self.stats.loss.append(totalLoss)
        return totalLoss, totalMSE, totalMaha

    def train(self, X, Y, JSystem_Y, model):
        X_train, X_val, Y_train, Y_val, J_train, J_val = self.trainTestSplit(X, Y, JSystem_Y)

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
            trainLoss, mseLoss, mahaLoss = self.__trainingPhase(
                X_train, Y_train, J_train, model, optimizer, lrScheduler
            )
            valLoss = self._validation_phase(X_val, Y_val, model)
            weights = self._softmax_weights().detach()

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: total={self._format_loss(trainLoss)}, "
                f"mse={self._format_loss(mseLoss)}, "
                f"maha={self._format_loss(mahaLoss)}  |  "
                f"Val: {self._format_loss(valLoss)}  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}  |  "
                f"w_mse={weights[0]:.3f}, w_maha={weights[1]:.3f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
class JVPMahanalobisTrainer(SchedulerMixin, SplitMixin, ValidationMixin, _statsSave):
    """
    MSE + Mahalanobis without pre-computed Jacobians.
    Uses JVP to compute the Mahalanobis-like penalty on the fly.

    loss = MSE + lossWeight * ||corrected_residual||^2
    """

    def __init__(
        self,
        testSize=0.25,
        lossWeight=1.08e-7,
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
        self.lossWeight = lossWeight

    @staticmethod
    def _jvp_mahalanobis_loss(J_diff: torch.Tensor) -> torch.Tensor:
        return (J_diff ** 2).sum(dim=1).mean()

    def __trainingPhase(self, X, Y, reactorModel, dt, model, optimizer, lrScheduler):
        model.train()
        totalLoss = totalMSE = totalMaha = 0.0

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            preds = model(bX)
            eps = preds.data - bY.data
            unscaled_error = model.inputScaler.J_1(bY) * eps
            J_scaling = model.outputScaler.J(model.outputScaler.inverse_transform(bY))
            target_phys = model.outputScaler.inverse_transform(bY).data
            J_diff = eps - dt * J_scaling * reactorModel.jvp(target_phys, None, unscaled_error)

            mseLoss  = nn.functional.mse_loss(preds.data, bY.data)
            mahaLoss = self.lossWeight * self._jvp_mahalanobis_loss(J_diff)
            loss = mseLoss + mahaLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrScheduler.step()

            n = bX.data.size(0)
            totalLoss += loss.item()     * n
            totalMSE  += mseLoss.item()  * n
            totalMaha += mahaLoss.item() * n

        N = X.nSamples
        totalLoss /= N; totalMSE /= N; totalMaha /= N
        self.stats.loss.append(totalLoss)
        return totalLoss, totalMSE, totalMaha

    def train(self, X, Y, reactorModel, dt, model):
        X_train, X_val, Y_train, Y_val = self.trainTestSplit(X, Y)

        model.to(self.trainingDevice).to(self.trainingDtype)
        optimizer, lrScheduler = self._build_optimizer(model)

        for epoch in range(self.epochs):
            trainLoss, mseLoss, mahaLoss = self.__trainingPhase(
                X_train, Y_train, reactorModel, dt, model, optimizer, lrScheduler
            )
            valLoss = self._validation_phase(X_val, Y_val, model)

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: total={self._format_loss(trainLoss)}, "
                f"mse={self._format_loss(mseLoss)}, "
                f"maha={self._format_loss(mahaLoss)}  |  "
                f"Val: {self._format_loss(valLoss)}  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

# TODO: essayer de changer la loss ||ypred-y||^2 + lambda ||J(ypred-y)||^2 en prefactor_for_scaling(function of lambda) * ||(lambda I + (1-lambda)J)(ypred-y)||^2.
#  avec lambda qui serait une fonction du nombre de conditionement moyen de la base de données. un objectif de +- court terme pourait être de montrer que le lambda optimal est corrélé au nombre de conditionement
#  en faisant une optimisation bayesienne qui porte uniquement sur lambda (coef exponetiel). on peut tracer alors val loss optimal de chaque base de donnée et son nombre moyen de conditionement (moyenne des log(conditionement))
#  Il doit bien y avoir un lien entre les deux quantités log.
#  essayer d'optimiser lambda dans prefactor_for_scaling(function of lambda) * ||(lambda I + (1-lambda)J)(ypred-y)||^2. tel que ça minimise la loss MSE sur les données (ou celle de val). ça resemblerai à une sorte de gradnorm
#  Si c'est trop dur, on peut aussi faire un annealing pour optimiser full mahanalobis au début et full MSE à la fin (mahanalobis s'optimise toujours assez facilement et permet de capter les zones raides. on vient ensuite capter les zones lentes avec mse.)

# TODO: étudier les différentes façons de scaler M(lambda) (soit normalisation élément par élément par la norme soit normalisation par la moyenne de la norme par batch, ça dépend si on veu conserver l'information de la rigidité absolue par rapport aux autre échatillons de la loss.
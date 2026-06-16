from POLENE_GPU.NN.trainer._statsSave import _statsSave
from POLENE_GPU.NN.trainer._mixins import SchedulerMixin, SplitMixin, ValidationMixin, EffectiveRankMixin, GradNormTrackingMixin, ActivationTrackingMixin, PerOutputGradMixin

import torch
import torch.nn as nn
from tqdm import tqdm
import gc


class logPenalizedTrainer(SchedulerMixin, SplitMixin, ValidationMixin, _statsSave, EffectiveRankMixin, GradNormTrackingMixin, ActivationTrackingMixin, PerOutputGradMixin):

    def __init__(
        self,
        testSize=0.25,
        lr=5e-3,
        warmupSteps=2000,
        decayRate=0.92,
        decaySteps=1500,
        epochs=50,
        batchSize=128,
        beta1=None,
        beta2=None,
        trackGradient=False,
        trackActivation=False,
        trackPerOutputGrad=False,
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
        self.trackGradient = trackGradient
        self.trackActivation = trackActivation
        self.trackPerOutputGrad = trackPerOutputGrad

        self.beta1 = beta1
        self.beta2 = beta2

    def __trainingPhase(self, X, Y, model, optimizer, lrScheduler, logScaler):
        model.train()
        totalLoss = 0.0
        totalRawLoss = 0.0
        totalLogLoss = 0.0

        if self.trackGradient:
            self._reset_grad_stats()
        if self.trackPerOutputGrad:
            self._reset_per_output_stats()

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            optimizer.zero_grad()

            if self.trackActivation:
                self._start_recording("train")
            preds = model(bX)
            if self.trackActivation:
                self._stop_recording()

            if self.trackPerOutputGrad:
                self._track_per_output_grads(model, preds.data, bY.data)

            lossRaw = nn.functional.mse_loss(preds.data, bY.data)

            preds_phys = model.outputScaler.inverse_transformTorch(preds.data)
            targets_phys = model.outputScaler.inverse_transformTorch(bY.data)

            eps = 1e-8
            log_pred = torch.log(preds_phys.abs().clamp(min=eps))
            log_target = torch.log(targets_phys.abs().clamp(min=eps))

            lossLog = nn.functional.mse_loss(log_pred, log_target)

            #lossLog = nn.functional.mse_loss(logScaler.transformTorch(scaler.inverse_transformTorch(preds.data)), logScaler.transformTorch(scaler.inverse_transformTorch(bY.data)))

            params = [p for p in model.parameters() if p.requires_grad]
            # Gradients de chaque loss séparément
            #g_mse = torch.autograd.grad(
            #    lossRaw, params,
            #    retain_graph=True,
            #    create_graph=False,
            #    allow_unused=True,
            #)
            #g_log = torch.autograd.grad(
            #    lossLog, params,
            #    retain_graph=True,
            #    create_graph=False,
            #    allow_unused=True,
            #)

            # Aplatit et concatène en un seul vecteur, ignore les None (allow_unused)
            #def flatten(grads):
            #    return torch.cat([
            #        g.detach().reshape(-1) for g in grads if g is not None
            #    ])

            #g1 = flatten(g_mse)
            #g2 = flatten(g_log)

            #norm1 = g1.norm()
            #norm2 = g2.norm()

            #ratio = norm2 / (norm1 + 1e-12)
            #cosine = torch.dot(g1, g2) / (norm1 * norm2 + 1e-12)
            #print(ratio, cosine)

            lamb = 0.1 #TODO: implement Gradnorm. il ne semble pas y avoir de cosine similarity negative sauf exception donc pas besoin de PCGrad
            loss = (1-lamb) * lossRaw + lamb * lossLog

            loss.backward()

            if self.trackGradient:
                self._track_grad_norms(model)  # If we want the gradient per step, we can chain rest, track and flush right here.
            optimizer.step()
            lrScheduler.step()

            totalLoss += loss.item() * bX.data.size(0)
            totalRawLoss += lossRaw.item() * bX.data.size(0)
            totalLogLoss += lossLog.item() * bX.data.size(0)

        self._flush_grad_epoch()
        self._flush_per_output_epoch()
        totalLoss /= X.nSamples
        totalRawLoss /= X.nSamples
        totalLogLoss /= X.nSamples
        self.stats.loss.append(totalLoss)
        return totalLoss, totalRawLoss, totalLogLoss

    def train(self, X, Y, model, logScaler):  #TODO: use precomputed Jacobian matrix computed with float 64 to prevent any data precision issue
        torch.cuda.empty_cache()
        gc.collect()

        X_train, X_val, Y_train, Y_val = self.trainTestSplit(X, Y)

        model.to(self.trainingDevice).to(self.trainingDtype)
        optimizer, lrScheduler = self._build_optimizer(model, self.beta1, self.beta2)

        self.stats.ranks.append(self.compute_effective_ranks(model))

        if self.trackActivation:
            self._register_activation_hooks(model)

        for epoch in range(self.epochs):
            if self.trackActivation:
                self._new_activation_epoch()

            trainLoss, rawLoss, logLoss = self.__trainingPhase(X_train, Y_train, model, optimizer, lrScheduler, logScaler)
            valLoss = self._validation_phase(X_val, Y_val, model)
            ranks = self.compute_effective_ranks(model)
            self.stats.ranks.append(ranks)
            if self.trackActivation:
                self._flush_activation_epoch()

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: {self._format_loss(trainLoss)} (raw: {self._format_loss(rawLoss)}, log: {self._format_loss(logLoss)})  |  "
                f"Val: {self._format_loss(valLoss)}  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if self.trackActivation:
            self._remove_activation_hooks()

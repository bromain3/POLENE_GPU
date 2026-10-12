from POLENE_GPU.NN.trainer._statsSave import _statsSave
from POLENE_GPU.NN.trainer._mixins import SchedulerMixin, SplitMixin, ValidationMixin, EffectiveRankMixin, GradNormTrackingMixin, ActivationTrackingMixin, PerOutputGradMixin

import torch
import torch.nn as nn
from tqdm import tqdm
import gc


class omegaTrainer(SchedulerMixin, SplitMixin, ValidationMixin, _statsSave, EffectiveRankMixin, GradNormTrackingMixin, ActivationTrackingMixin, PerOutputGradMixin):

    def __init__(
        self,
        testSize=0.25,
        lr=5e-3,
        warmupSteps=2000,
        decayRate=0.92,
        decaySteps=1500,
        epochs=50,
        batchSize=128,
        trainingNoise=0.0,
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
        self.trainingNoise = trainingNoise

        self.beta1 = beta1
        self.beta2 = beta2

    def __trainingPhase(self, X, Y, inc, model, optimizer, lrScheduler, sampleWeights=None):
        model.train()
        totalLoss = 0.0

        if self.trackGradient:
            self._reset_grad_stats()
        if self.trackPerOutputGrad:
            self._reset_per_output_stats()

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype).data
            bX = bX + torch.randn_like(bX) * self.trainingNoise
            bY = Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            binc = inc[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            optimizer.zero_grad()

            if self.trackActivation:
                self._start_recording("train")

            if model.predictState:
                #TODO: vastly improve that by using all precomputed values and descale only the non bst scaler
                states = model(bX, tensorInput=True)
                states_phys = model.inputScaler.inverse_transformTorch(states)
                X_phys = model.inputScaler.inverse_transformTorch(bX)
                preds = model.chemStdScaler.transformTorch(model.chemScaler.transformTorch(states_phys) - model.chemScaler.transformTorch(X_phys))



            else:
                raise ValueError('this trainer only suports predictState')

            if self.trackActivation:
                self._stop_recording()

            if self.trackPerOutputGrad:
                self._track_per_output_grads(model, preds, bY.data)

            # --- CALCUL DE LA LOSS PONDÉRÉE ---
            if sampleWeights is not None:
                # On récupère le batch de poids
                bW = sampleWeights[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
                # On calcule la MSE par élément sans réduction
                raw_loss = nn.functional.mse_loss(preds, bY.data, reduction='none')
                # On applique les poids et on fait la moyenne
                # bW doit avoir une forme compatible pour le broadcast (ex: [batch, 1] ou [batch])
                loss = (raw_loss * bW.data).mean()
            else:
                loss_inc = nn.functional.mse_loss(preds, binc.data)
                #loss_states = nn.functional.mse_loss(states, bY.data)
                loss = loss_inc# + loss_states

            loss.backward()

            #total_norm = 0

            #for param in model.parameters():
            #    if param.grad is not None:
            #        total_norm += param.grad.norm() ** 2

            #total_norm = total_norm ** 0.5

            #print("Gradient global :", total_norm.item())

            if self.trackGradient:
                self._track_grad_norms(model)  # If we want the gradient per step, we can chain rest, track and flush right here.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            lrScheduler.step()

            totalLoss += loss.item() * bX.data.size(0)

        self._flush_grad_epoch()
        self._flush_per_output_epoch()
        totalLoss /= X.nSamples
        self.stats.loss.append(totalLoss)
        return totalLoss

    def train(self, X, Y, inc, model, sampleWeights=None):
        torch.cuda.empty_cache()
        gc.collect()

        if sampleWeights is not None:
            X_train, X_val, Y_train, Y_val, inc_train, inc_val, sampleWeights_train, sampleWeights_val = self.trainTestSplit(X, Y, inc, sampleWeights)
        else:
            X_train, X_val, Y_train, Y_val, inc_train, inc_val = self.trainTestSplit(X, Y, inc)
            sampleWeights_train, sampleWeights_val = None, None

        model.to(self.trainingDevice).to(self.trainingDtype)
        optimizer, lrScheduler = self._build_optimizer(model, self.beta1, self.beta2)

        self.stats.ranks.append(self.compute_effective_ranks(model))

        if self.trackActivation:
            self._register_activation_hooks(model)

        for epoch in range(self.epochs):
            if self.trackActivation:
                self._new_activation_epoch()

            trainLoss = self.__trainingPhase(X_train, Y_train, inc_train, model, optimizer, lrScheduler, sampleWeights_train)
            valLoss = self._validation_phase(X_val, Y_val, model, sampleWeights_val)
            ranks = self.compute_effective_ranks(model)
            self.stats.ranks.append(ranks)
            if self.trackActivation:
                self._flush_activation_epoch()

            print(
                f"[{epoch + 1}/{self.epochs}] "
                f"Train: {self._format_loss(trainLoss)}  |  "
                f"Val: {self._format_loss(valLoss)}  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if self.trackActivation:
            self._remove_activation_hooks()

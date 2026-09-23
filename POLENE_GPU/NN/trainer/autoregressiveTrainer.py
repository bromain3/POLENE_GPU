from POLENE_GPU.NN.trainer._statsSave import _statsSave
from POLENE_GPU.NN.trainer._mixins import SchedulerMixin, SplitMixin, ValidationMixin, EffectiveRankMixin, GradNormTrackingMixin, ActivationTrackingMixin, PerOutputGradMixin

import torch
import torch.nn as nn
from tqdm import tqdm
import gc


class autoregressiveTrainer(SchedulerMixin, SplitMixin, ValidationMixin, _statsSave, EffectiveRankMixin, GradNormTrackingMixin, ActivationTrackingMixin, PerOutputGradMixin):

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

    def autoregLoss(self, model, bX, bY):
        seq_len = bY.shape[1]

        pred = bX
        preds = []
        for _ in range(seq_len):
            if self.trackActivation:
                self._start_recording("train")
            pred = model(pred)
            if self.trackActivation:
                self._stop_recording()
            preds.append(pred.data)
        preds = torch.stack(preds, dim=1)

        if self.trackPerOutputGrad:
            self._track_per_output_grads(model, preds, bY)

        loss = nn.functional.mse_loss(preds, bY)
        return(loss)

    def __trainingPhase(self, X, Ys, model, optimizer, lrScheduler):
        model.train()
        totalLoss = 0.0

        if self.trackGradient:
            self._reset_grad_stats()
        if self.trackPerOutputGrad:
            self._reset_per_output_stats()

        for start in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Ys[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            optimizer.zero_grad()

            loss = self.autoregLoss(model, bX, bY)

            loss.backward()

            if self.trackGradient:
                self._track_grad_norms(model)  # If we want the gradient per step, we can chain rest, track and flush right here.
            optimizer.step()
            lrScheduler.step()

            totalLoss += loss.detach() * bX.data.size(0)

        self._flush_grad_epoch()
        self._flush_per_output_epoch()
        totalLoss =  (totalLoss / X.nSamples).item()
        self.stats.loss.append(totalLoss)
        return totalLoss

    def trainTestSplit(self, X, Ys, samp=None):
        """
        Split X, Y and any number of extra datasets (e.g. J) with the same
        random permutation.

        Args:
            X, Y    : datasets with a .nSamples attribute and __getitem__
            *extras : any additional datasets that must be split identically

        Returns:
            X_train, X_test, Y_train, Y_test [, extra_train, extra_test, ...]
        """
        all_ds = [X] + Ys
        n_samples = X.nSamples
        for ds in all_ds:
            if ds.nSamples != n_samples:
                raise ValueError(
                    "All datasets passed to trainTestSplit must have the same number of samples."
                )

        n_test = int(n_samples * self.testSize)
        perm = torch.randperm(n_samples)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

        result = []
        for ds in all_ds:
            result.append(ds[train_idx].contiguous())
            result.append(ds[test_idx].contiguous())

        Ys_train = []
        Ys_test = []
        for i in range(2, len(result), 2):
            Ys_train.append(result[i])
            Ys_test.append(result[i + 1])

        return (result[0], result[1], Ys_train, Ys_test)

    def train(self, X, Ys, model):
        torch.cuda.empty_cache()
        gc.collect()

        X.data = X.data.pin_memory()
        for y in Ys:
            y.data = y.data.pin_memory()

        X_train, X_val, Ys_train, Ys_val = self.trainTestSplit(X, Ys)
        Ys_train = torch.stack([yy.data for yy in Ys_train], dim=1)
        Ys_val = torch.stack([yy.data for yy in Ys_val], dim=1)


        model.to(self.trainingDevice).to(self.trainingDtype)
        optimizer, lrScheduler = self._build_optimizer(model, self.beta1, self.beta2)

        self.stats.ranks.append(self.compute_effective_ranks(model))

        if self.trackActivation:
            self._register_activation_hooks(model)

        for epoch in range(self.epochs):
            if self.trackActivation:
                self._new_activation_epoch()

            trainLoss = self.__trainingPhase(X_train, Ys_train, model, optimizer, lrScheduler)
            valLoss = self._validation_phase(X_val, Ys_val[:, 0, :], model)
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

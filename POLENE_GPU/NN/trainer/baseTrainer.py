from POLENE_GPU.NN.trainer._statsSave import _statsSave

import torch
import torch.nn as nn
from tqdm import tqdm
import gc

class baseTrainer(_statsSave):
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

    def __trainingPhase(self, X, Y, model, optimizer, lrScheduler, lossFunc):
        model.train()
        totalLoss = 0

        for batch in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            batch_X = X[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_Y = Y[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = lossFunc(preds.data, batch_Y.data)
            loss.backward()
            optimizer.step()
            lrScheduler.step()
            totalLoss += loss.item() * batch_X.data.size(0)

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
            trainingLoss = self.__trainingPhase(X, Y, model, optimizer, lrScheduler, lossFunc)
            validationLoss = self.__validationPhase(Xval, Yval, model, optimizer, lrScheduler, lossFunc)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{epoch + 1}/{self.epochs}] "
                  f"Train: {self._format_loss(trainingLoss)}  |  "
                  f"Val: {self._format_loss(validationLoss)}  |  "
                  f"LR: {current_lr:.2e}")


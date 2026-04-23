from POLENE_GPU.NN.trainer._statsSave import _statsSave

import torch
import torch.nn as nn
from tqdm import tqdm
import gc

class mahanalobisTrainerGradNorm(_statsSave):
    """
    Utilisation de gradnorm pour ajuster les coeficients de pondération entre la loss pur mse et la loss pur mahanalobis
    """
    def __init__(self, testSize=0.25, lossWeight=2.5e-7, emaBeta=0.99, lr=5e-3, warmupSteps=2000, decayRate=0.92, decaySteps=1500, epochs=50, batchSize=128, trainingDevice=None, trainingDtype=None):
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
        self.taskWeights = torch.nn.Parameter(torch.ones(2, device=self.trainingDevice))
        self.weightOptimizer = torch.optim.Adam([self.taskWeights], lr=1e-3)
        self.init_mse = None
        self.init_mah = None
        self.alpha = 0.5

        self.emaMse = None
        self.emaMah = None
        self.emaBeta = emaBeta

    def warmup_exp_decay(self, k):
        if self.warmupSteps > 0 and k < self.warmupSteps:
            return (k + 1) / self.warmupSteps
        return self.decayRate ** ((k - self.warmupSteps) / self.decaySteps)

    def trainTestSplit(self, X, Y, J):
        if X.nSamples != Y.nSamples or X.nSamples != J.nSamples:
            raise ValueError("X, Y and J must have the same number of samples")

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
        J_train = J[train_idx].contiguous()
        J_test = J[test_idx].contiguous()

        return X_train, X_test, Y_train, Y_test, J_train, J_test

    @staticmethod
    def mahalanobis_loss(preds, targets, J):
        diff = preds - targets  # (B, d)
        diff = diff.unsqueeze(-1)  # (B, d, 1)

        J_diff = torch.bmm(J, diff)  # (B, d, 1)
        loss = (J_diff ** 2).sum(dim=(1, 2))  # (B,)

        return loss.mean()

    @staticmethod
    def grad_norm(loss, params):
        grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
        norm = torch.norm(torch.stack([g.norm() for g in grads if g is not None]))
        return norm

    def update_ema(self, mseLoss, mahLoss):
        mse = mseLoss.detach()
        mah = mahLoss.detach()

        if self.emaMse is None:
            self.emaMse = mse
            self.emaMah = mah
        else:
            self.emaMse = self.emaBeta * self.emaMse + (1 - self.emaBeta) * mse
            self.emaMah = self.emaBeta * self.emaMah + (1 - self.emaBeta) * mah

        mean = (self.emaMse + self.emaMah) / 2.0

        r_mse = self.emaMse / mean
        r_mah = self.emaMah / mean

        return r_mse, r_mah

    def __trainingPhase(self, X, Y, J_Y, model, optimizer, lrScheduler, lossFunc):
        model.train()
        totalLoss = 0
        totalMSELoss = 0
        totalMahalanobisLoss = 0

        for batch in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            batch_X = X[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_Y = Y[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_J_Y = J_Y[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            preds = model(batch_X)
            mseLoss = lossFunc(preds.data, batch_Y.data)
            mahalanobisLoss = self.lossWeight * self.mahalanobis_loss(preds.data, batch_Y.data, batch_J_Y.data)
            weights = torch.softmax(self.taskWeights, dim=0)
            weightedMSE = weights[0] * mseLoss
            weightedMAH = weights[1] * mahalanobisLoss

            loss = weightedMSE + weightedMAH

            # --- GradNorm part ---
            shared_params = [p for p in model.parameters() if p.requires_grad]
            #if self.init_mse is None:
            #    self.init_mse = mseLoss.detach()
            #    self.init_mah = mahalanobisLoss.detach()

            G1 = self.grad_norm(weightedMSE, shared_params)
            G2 = self.grad_norm(weightedMAH, shared_params)

            G_avg = (G1 + G2) / 2

            # training rate (approximation simple)
            #L1_ratio = mseLoss / self.init_mse
            #L2_ratio = mahalanobisLoss / self.init_mah

            #mean = (L1_ratio + L2_ratio) / 2

            #r1 = L1_ratio / mean
            #r2 = L2_ratio / mean
            r1, r2 = self.update_ema(mseLoss, mahalanobisLoss)

            gradLoss = torch.abs(G1 - G_avg * (r1 ** self.alpha)) + \
                       torch.abs(G2 - G_avg * (r2 ** self.alpha))

            # --- update weights ---
            self.weightOptimizer.zero_grad()
            gradLoss.backward(retain_graph=True)
            self.weightOptimizer.step()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lrScheduler.step()
            totalLoss += loss.item() * batch_X.data.size(0)
            totalMSELoss += mseLoss * batch_X.data.size(0)
            totalMahalanobisLoss += mahalanobisLoss * batch_X.data.size(0)

        totalLoss /= X.nSamples
        totalMSELoss /= X.nSamples
        totalMahalanobisLoss /= X.nSamples
        self.stats.loss.append(totalLoss)

        return totalLoss, totalMSELoss, totalMahalanobisLoss, weights[0], weights[1]

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


    def train(self, X, Y, JSystem_Y, model):
        X, Xval, Y, Yval, JSystem_Y, JSystemVal_Y = self.trainTestSplit(X, Y, JSystem_Y)
        #  data is already shuffled.

        model.to(self.trainingDevice).to(self.trainingDtype)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[self.warmup_exp_decay])
        lossFunc = nn.MSELoss()

        for epoch in range(self.epochs):
            trainingLoss, mseLoss, mahaLoss, lambdamse, lambdamaha = self.__trainingPhase(X, Y, JSystem_Y, model, optimizer, lrScheduler, lossFunc)
            validationLoss = self.__validationPhase(Xval, Yval, model, optimizer, lrScheduler, lossFunc)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{epoch + 1}/{self.epochs}] "
                  f"Train: total loss: {self._format_loss(trainingLoss)}, mse loss: {self._format_loss(mseLoss)}, maha loss: {self._format_loss(mahaLoss)} |  "
                  f"Val: {self._format_loss(validationLoss)}  |  "
                  f"LR: {current_lr:.2e}  |  "
                  f"mse weight: {self._format_loss(lambdamse)}, maha weight: {self._format_loss(lambdamaha)}")



class jacobianMahanalobisTrainer(_statsSave):
    """
    weighting entre loss pur mse et loss pur mahanalobis, nécessite précalcul de la jacobienne
    """
    def __init__(self, testSize=0.25, lossWeight=1.08e-7, lr=5e-3, warmupSteps=2000, decayRate=0.92, decaySteps=1500, epochs=50, batchSize=128, trainingDevice=None, trainingDtype=None):
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

    def warmup_exp_decay(self, k):
        if self.warmupSteps > 0 and k < self.warmupSteps:
            return (k + 1) / self.warmupSteps
        return self.decayRate ** ((k - self.warmupSteps) / self.decaySteps)

    def trainTestSplit(self, X, Y, J):
        if X.nSamples != Y.nSamples or X.nSamples != J.nSamples:
            raise ValueError("X, Y and J must have the same number of samples")

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
        J_train = J[train_idx].contiguous()
        J_test = J[test_idx].contiguous()

        return X_train, X_test, Y_train, Y_test, J_train, J_test

    @staticmethod
    def mahalanobis_loss(preds, targets, J):
        diff = preds - targets  # (B, d)
        diff = diff.unsqueeze(-1)  # (B, d, 1)

        J_diff = torch.bmm(J, diff)  # (B, d, 1)
        loss = (J_diff ** 2).sum(dim=(1, 2))  # (B,)

        return loss.mean()

    def __trainingPhase(self, X, Y, J_Y, model, optimizer, lrScheduler, lossFunc):
        model.train()
        totalLoss = 0
        totalMSELoss = 0
        totalMahalanobisLoss = 0

        for batch in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            batch_X = X[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_Y = Y[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_J_Y = J_Y[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            preds = model(batch_X)
            mseLoss = lossFunc(preds.data, batch_Y.data)
            mahalanobisLoss = self.lossWeight * self.mahalanobis_loss(preds.data, batch_Y.data, batch_J_Y.data)
            loss = mseLoss + mahalanobisLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lrScheduler.step()
            totalLoss += loss.item() * batch_X.data.size(0)
            totalMSELoss += mseLoss * batch_X.data.size(0)
            totalMahalanobisLoss += mahalanobisLoss * batch_X.data.size(0)

        totalLoss /= X.nSamples
        totalMSELoss /= X.nSamples
        totalMahalanobisLoss /= X.nSamples
        self.stats.loss.append(totalLoss)

        return totalLoss, totalMSELoss, totalMahalanobisLoss

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


    def train(self, X, Y, JSystem_Y, model):
        X, Xval, Y, Yval, JSystem_Y, JSystemVal_Y = self.trainTestSplit(X, Y, JSystem_Y)
        #  data is already shuffled.

        model.to(self.trainingDevice).to(self.trainingDtype)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[self.warmup_exp_decay])
        lossFunc = nn.MSELoss()

        for epoch in range(self.epochs):
            trainingLoss, mseLoss, mahaLoss = self.__trainingPhase(X, Y, JSystem_Y, model, optimizer, lrScheduler, lossFunc)
            validationLoss = self.__validationPhase(Xval, Yval, model, optimizer, lrScheduler, lossFunc)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{epoch + 1}/{self.epochs}] "
                  f"Train: total loss: {self._format_loss(trainingLoss)}, mse loss: {self._format_loss(mseLoss)}, maha loss: {self._format_loss(mahaLoss)} |  "
                  f"Val: {self._format_loss(validationLoss)}  |  "
                  f"LR: {current_lr:.2e}")


class JVPMahanalobisTrainer(_statsSave):
    """
    weighting entre loss pur mse et loss pur mahanalobis, sans précalcul de la jacobienne
    """
    def __init__(self, testSize=0.25, lossWeight=1.08e-7, lr=5e-3, warmupSteps=2000, decayRate=0.92, decaySteps=1500, epochs=50, batchSize=128, trainingDevice=None, trainingDtype=None):
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
    def mahalanobis_loss(J_diff):
        return(J_diff ** 2).sum(dim=1).mean() # (B,)

    def __trainingPhase(self, X, Y, reactorModel, dt, model, optimizer, lrScheduler, lossFunc):
        model.train()
        totalLoss = 0
        totalMSELoss = 0
        totalMahalanobisLoss = 0

        for batch in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            batch_X = X[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_Y = Y[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            preds = model(batch_X)

            eps = (preds.data - batch_Y.data)
            unscaled_error = model.inputScaler.J_1(batch_Y) * eps
            J_scaling = model.outputScaler.J(model.outputScaler.inverse_transform(batch_Y))
            target_phys = model.outputScaler.inverse_transform(batch_Y).data
            J_diff = eps - dt * J_scaling * reactorModel.jvp(target_phys, None, unscaled_error)

            mseLoss = lossFunc(preds.data, batch_Y.data)
            mahalanobisLoss = self.lossWeight * self.mahalanobis_loss(J_diff)
            loss = mseLoss + mahalanobisLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lrScheduler.step()
            totalLoss += loss.item() * batch_X.data.size(0)
            totalMSELoss += mseLoss * batch_X.data.size(0)
            totalMahalanobisLoss += mahalanobisLoss * batch_X.data.size(0)

        totalLoss /= X.nSamples
        totalMSELoss /= X.nSamples
        totalMahalanobisLoss /= X.nSamples
        self.stats.loss.append(totalLoss)

        return totalLoss, totalMSELoss, totalMahalanobisLoss

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


    def train(self, X, Y, reactorModel, dt, model):
        X, Xval, Y, Yval = self.trainTestSplit(X, Y)
        #  data is already shuffled.

        model.to(self.trainingDevice).to(self.trainingDtype)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[self.warmup_exp_decay])
        lossFunc = nn.MSELoss()

        for epoch in range(self.epochs):
            trainingLoss, mseLoss, mahaLoss = self.__trainingPhase(X, Y, reactorModel, dt, model, optimizer, lrScheduler, lossFunc)
            validationLoss = self.__validationPhase(Xval, Yval, model, optimizer, lrScheduler, lossFunc)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{epoch + 1}/{self.epochs}] "
                  f"Train: total loss: {self._format_loss(trainingLoss)}, mse loss: {self._format_loss(mseLoss)}, maha loss: {self._format_loss(mahaLoss)} |  "
                  f"Val: {self._format_loss(validationLoss)}  |  "
                  f"LR: {current_lr:.2e}")


class mahanalobisTrainer(_statsSave):
    """
    loss de mahanalobis avec M = (1-lambda) I + lambda J qui permet d'ajuster la compression / decompression des distances
    The prefactor 1.08e-7 commes from the gradnorm trainer. this value ensures the pure mahanalobis loss learns as fast as the pure MSE with no prefactor
    """
    def __init__(self, testSize=0.25, lamb=5, lr=5e-3, warmupSteps=2000, decayRate=0.92, decaySteps=1500, epochs=50, batchSize=128, trainingDevice=None, trainingDtype=None):
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

    def warmup_exp_decay(self, k):
        if self.warmupSteps > 0 and k < self.warmupSteps:
            return (k + 1) / self.warmupSteps
        return self.decayRate ** ((k - self.warmupSteps) / self.decaySteps)

    def trainTestSplit(self, X, Y, J):
        if X.nSamples != Y.nSamples or X.nSamples != J.nSamples:
            raise ValueError("X, Y and J must have the same number of samples")

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
        J_train = J[train_idx].contiguous()
        J_test = J[test_idx].contiguous()

        return X_train, X_test, Y_train, Y_test, J_train, J_test

    def mahalanobis_loss(self, preds, targets, J):
        diff = preds - targets  # (B, d)
        diff = diff.unsqueeze(-1)  # (B, d, 1)
        # Construction de la matrice M
        I = torch.eye(diff.shape[1], device=self.trainingDevice, dtype=self.trainingDtype).unsqueeze(0)
        M = (1 - self.lamb) * I + self.lamb * J  # (B, d, d)

        # TODO: add option to normalize or not the matrix as it changes a bit the results (normalizing sample per sample is not equivalent to normalizing globaly on the batch)
        #norms = torch.linalg.norm(M, ord='fro', dim=(1, 2))  # Shape: (B,)
        #norms = norms.unsqueeze(-1).unsqueeze(-1)
        #M_scaled = diff.shape[1] ** (1/2) * M / norms

        J_diff = torch.bmm(M, diff)  # (B, d, 1)
        loss = (J_diff ** 2).mean(dim=(1, 2))  # (B,)
        return loss.mean()

    def __trainingPhase(self, X, Y, J_Y, model, optimizer, lrScheduler, lossFunc):
        model.train()
        totalLoss = 0

        for batch in tqdm(range(0, X.nSamples, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]['lr'])

            batch_X = X[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_Y = Y[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            batch_J_Y = J_Y[batch:batch + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            preds = model(batch_X)

            #prefactor = ((1 - self.lamb) * 1.0 + self.lamb * self.mahaPrefactor) ** 2  # TODO: normalize the jacobian matrix for removing the need for prefactor ?
            # TODO: étudier le scaling de la matrice de mahanalobis pas pour chaque échantillon mais par batch, comme ça plus besoin de prefactor
            mahalanobisLoss = self.mahalanobis_loss(preds.data, batch_Y.data, batch_J_Y.data)  # prefactor * self.mahalanobis_loss(preds.data, batch_Y.data, batch_J_Y.data)

            optimizer.zero_grad()
            mahalanobisLoss.backward()
            optimizer.step()

            lrScheduler.step()
            totalLoss += mahalanobisLoss.item() * batch_X.data.size(0)

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


    def train(self, X, Y, JSystem_Y, model):
        torch.cuda.empty_cache()
        gc.collect()

        X, Xval, Y, Yval, JSystem_Y, JSystemVal_Y = self.trainTestSplit(X, Y, JSystem_Y)
        #  data is already shuffled.

        model.to(self.trainingDevice).to(self.trainingDtype)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[self.warmup_exp_decay])
        lossFunc = nn.MSELoss()

        for epoch in range(self.epochs):
            trainingLoss = self.__trainingPhase(X, Y, JSystem_Y, model, optimizer, lrScheduler, lossFunc)
            validationLoss = self.__validationPhase(Xval, Yval, model, optimizer, lrScheduler, lossFunc)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{epoch + 1}/{self.epochs}] "
                  f"Train: {self._format_loss(trainingLoss)}  |  "
                  f"Val: {self._format_loss(validationLoss)}  |  "
                  f"LR: {current_lr:.2e}")

# TODO: essayer de changer la loss ||ypred-y||^2 + lambda ||J(ypred-y)||^2 en prefactor_for_scaling(function of lambda) * ||(lambda I + (1-lambda)J)(ypred-y)||^2.
#  avec lambda qui serait une fonction du nombre de conditionement moyen de la base de données. un objectif de +- court terme pourait être de montrer que le lambda optimal est corrélé au nombre de conditionement
#  en faisant une optimisation bayesienne qui porte uniquement sur lambda (coef exponetiel). on peut tracer alors val loss optimal de chaque base de donnée et son nombre moyen de conditionement (moyenne des log(conditionement))
#  Il doit bien y avoir un lien entre les deux quantités log.
#  essayer d'optimiser lambda dans prefactor_for_scaling(function of lambda) * ||(lambda I + (1-lambda)J)(ypred-y)||^2. tel que ça minimise la loss MSE sur les données (ou celle de val). ça resemblerai à une sorte de gradnorm
#  Si c'est trop dur, on peut aussi faire un annealing pour optimiser full mahanalobis au début et full MSE à la fin (mahanalobis s'optimise toujours assez facilement et permet de capter les zones raides. on vient ensuite capter les zones lentes avec mse.)

# TODO: étudier les différentes façons de scaler M(lambda) (soit normalisation élément par élément par la norme soit normalisation par la moyenne de la norme par batch, ça dépend si on veu conserver l'information de la rigidité absolue par rapport aux autre échatillons de la loss.
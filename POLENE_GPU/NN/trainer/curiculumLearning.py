import math
import gc
from types import SimpleNamespace

import torch
import torch.nn as nn
from tqdm import tqdm

from POLENE_GPU.dataHandler.serie import concat


class MixScheduler:
    """
    Calcule la proportion d'échantillons issus du dataset A (référence)
    à utiliser à chaque époque, en interpolant entre startRatio et endRatio.

    ratio(epoch) -> proportion de A ; (1 - ratio) -> proportion de B
    """

    def __init__(self, startRatio, endRatio, totalEpochs, mode="linear"):
        assert 0.0 <= startRatio <= 1.0 and 0.0 <= endRatio <= 1.0
        self.startRatio = startRatio
        self.endRatio = endRatio
        self.totalEpochs = max(totalEpochs - 1, 1)
        self.mode = mode

    def ratio(self, epoch):
        t = min(epoch / self.totalEpochs, 1.0)
        if self.mode == "linear":
            frac = t
        elif self.mode == "cosine":
            frac = (1 - math.cos(math.pi * t)) / 2
        else:
            raise ValueError(f"mode inconnu: {self.mode}")
        return self.startRatio + (self.endRatio - self.startRatio) * frac


def _raw(x):
    """Extrait le tenseur brut d'un objet `serie` (ou le renvoie tel quel si
    c'est déjà un tenseur / n'a pas d'attribut .data)."""
    return x.data if hasattr(x, "data") else x


class MixTrainer:
    """
    Trainer standalone (aucune dépendance à baseTrainer) qui entraîne un
    modèle sur un mélange de deux bases de données (A = référence,
    B = perturbée), avec une proportion A/B qui évolue à chaque époque.

    X et Y sont ici des objets `serie` (cf. ton module de données) :
    - indexation par int / slice / list / tensor -> renvoie une nouvelle `serie`
    - .nSamples, .to(device), .data (tenseur brut sous-jacent)
    - concaténation via la fonction top-level `concat([...])` (nécessite les
      mêmes colName sur tous les éléments)
    """

    def __init__(
        self,
        startRatio=0.5,
        endRatio=0.9,
        mixMode="linear",
        samplesPerEpoch=None,
        testSize=0.25,
        lr=5e-3,
        warmupSteps=2000,
        decayRate=0.92,
        decaySteps=1500,
        epochs=50,
        batchSize=128,
        beta1=0.9,
        beta2=0.999,
        trainingDevice=None,
        trainingDtype=torch.float32,
    ):
        self.startRatio = startRatio
        self.endRatio = endRatio
        self.mixMode = mixMode
        self.samplesPerEpoch = samplesPerEpoch

        self.testSize = testSize
        self.lr = lr
        self.warmupSteps = warmupSteps
        self.decayRate = decayRate
        self.decaySteps = decaySteps
        self.epochs = epochs
        self.batchSize = batchSize
        self.beta1 = beta1
        self.beta2 = beta2
        self.trainingDevice = trainingDevice or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trainingDtype = trainingDtype

        self.stats = SimpleNamespace(loss=[], valLossA=[], valLossB=[], lr=[], mixRatio=[])
        self._scheduler = None

    # ------------------------------------------------------------------
    # Split train/val (aléatoire, sans stratification) sur des `serie`
    # ------------------------------------------------------------------
    def _train_test_split(self, X, Y, W=None):
        n = X.nSamples
        idx = torch.randperm(n)
        nVal = int(n * self.testSize)
        valIdx, trainIdx = idx[:nVal], idx[nVal:]

        X_train, Y_train = X[trainIdx], Y[trainIdx]
        X_val, Y_val = X[valIdx], Y[valIdx]
        W_train = W[trainIdx] if W is not None else None
        W_val = W[valIdx] if W is not None else None
        return X_train, X_val, Y_train, Y_val, W_train, W_val

    # ------------------------------------------------------------------
    # Optimiseur + scheduler LR (warmup linéaire puis décroissance exp.)
    # ------------------------------------------------------------------
    def _build_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        def lr_lambda(step):
            if step < self.warmupSteps:
                return step / max(self.warmupSteps, 1)
            decaySteps = max((step - self.warmupSteps) / self.decaySteps, 0)
            return self.decayRate ** decaySteps

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    @staticmethod
    def _format_loss(val):
        return f"{val:.3e}" if val < 1e-3 else f"{val:.6f}"

    # ------------------------------------------------------------------
    # Construction du batch d'époque (mélange A/B selon le ratio courant)
    # ------------------------------------------------------------------
    def _build_epoch_batch(self, XA, YA, XB, YB, ratio, WA=None, WB=None):
        nTotal = self.samplesPerEpoch or XA.nSamples
        nA = int(round(nTotal * ratio))
        nB = max(nTotal - nA, 0)

        idxA = torch.randperm(XA.nSamples)[:nA]
        idxB = torch.randperm(XB.nSamples)[:nB]

        XaS, YaS = XA[idxA], YA[idxA]
        XbS, YbS = XB[idxB], YB[idxB]
        WaS = WA[idxA] if WA is not None else None
        WbS = WB[idxB] if WB is not None else None

        Xep = concat([XaS, XbS])
        Yep = concat([YaS, YbS])
        Wep = concat([WaS, WbS]) if (WA is not None and WB is not None) else None

        perm = torch.randperm(Xep.nSamples)
        Xep, Yep = Xep[perm], Yep[perm]
        Wep = Wep[perm] if Wep is not None else None
        return Xep, Yep, Wep

    # ------------------------------------------------------------------
    # Une époque d'entraînement
    # ------------------------------------------------------------------
    def _training_phase(self, X, Y, model, optimizer, lrScheduler, sampleWeights=None):
        model.train()
        totalLoss = 0.0
        n = X.nSamples

        for start in tqdm(range(0, n, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]["lr"])

            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)

            optimizer.zero_grad()
            preds = model(bX)

            if sampleWeights is not None:
                bW = sampleWeights[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
                raw_loss = nn.functional.mse_loss(_raw(preds), _raw(bY), reduction="none")
                loss = (raw_loss * _raw(bW)).mean()
            else:
                loss = nn.functional.mse_loss(_raw(preds), _raw(bY))

            loss.backward()
            optimizer.step()
            lrScheduler.step()

            totalLoss += loss.item() * bX.nSamples

        totalLoss /= n
        return totalLoss

    # ------------------------------------------------------------------
    # Validation (sur un seul dataset à la fois)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validation_phase(self, X, Y, model, sampleWeights=None):
        model.eval()
        totalLoss = 0.0
        n = X.nSamples

        for start in range(0, n, self.batchSize):
            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            preds = model(bX)

            if sampleWeights is not None:
                bW = sampleWeights[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
                raw_loss = nn.functional.mse_loss(_raw(preds), _raw(bY), reduction="none")
                loss = (raw_loss * _raw(bW)).mean()
            else:
                loss = nn.functional.mse_loss(_raw(preds), _raw(bY))

            totalLoss += loss.item() * bX.nSamples

        return totalLoss / n

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------
    def train(self, XA, YA, XB, YB, model, sampleWeightsA=None, sampleWeightsB=None):
        torch.cuda.empty_cache()
        gc.collect()

        XA_train, XA_val, YA_train, YA_val, WA_train, WA_val = self._train_test_split(XA, YA, sampleWeightsA)
        XB_train, XB_val, YB_train, YB_val, WB_train, WB_val = self._train_test_split(XB, YB, sampleWeightsB)

        model.to(self.trainingDevice).to(self.trainingDtype)
        optimizer, lrScheduler = self._build_optimizer(model)
        self._scheduler = MixScheduler(self.startRatio, self.endRatio, self.epochs, self.mixMode)

        for epoch in range(self.epochs):
            ratio = self._scheduler.ratio(epoch)
            self.stats.mixRatio.append(ratio)

            Xep, Yep, Wep = self._build_epoch_batch(
                XA_train, YA_train, XB_train, YB_train, ratio, WA_train, WB_train
            )

            trainLoss = self._training_phase(Xep, Yep, model, optimizer, lrScheduler, Wep)
            valLossA = self._validation_phase(XA_val, YA_val, model, WA_val)
            valLossB = self._validation_phase(XB_val, YB_val, model, WB_val)

            self.stats.loss.append(trainLoss)
            self.stats.valLossA.append(valLossA)
            self.stats.valLossB.append(valLossB)

            print(
                f"[{epoch + 1}/{self.epochs}] ratio A/B: {ratio:.2f}/{1 - ratio:.2f} | "
                f"Train: {self._format_loss(trainLoss)} | "
                f"Val A: {self._format_loss(valLossA)} | Val B: {self._format_loss(valLossB)} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        return model
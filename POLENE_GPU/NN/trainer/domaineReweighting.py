import math
import gc
from types import SimpleNamespace

import torch
import torch.nn as nn
from tqdm import tqdm

from POLENE_GPU.dataHandler.serie import concat


class WeightScheduler:
    """
    Interpole un scalaire (ex: le poids de la loss sur la base B) entre
    startValue et endValue au fil des époques.

    mode: "linear", "cosine", "exponential"
      - "exponential" : décroissance géométrique startValue * decayRate**epoch,
        clampée à endValue (pratique pour une décroissance "vite au début,
        lentement à la fin").
    """

    def __init__(self, startValue, endValue, totalEpochs, mode="linear", decayRate=0.95):
        self.startValue = startValue
        self.endValue = endValue
        self.totalEpochs = max(totalEpochs - 1, 1)
        self.mode = mode
        self.decayRate = decayRate

    def value(self, epoch):
        if self.mode == "exponential":
            v = self.startValue * (self.decayRate ** epoch)
            return max(v, self.endValue) if self.endValue <= self.startValue else min(v, self.endValue)

        t = min(epoch / self.totalEpochs, 1.0)
        if self.mode == "linear":
            frac = t
        elif self.mode == "cosine":
            frac = (1 - math.cos(math.pi * t)) / 2
        else:
            raise ValueError(f"mode inconnu: {self.mode}")
        return self.startValue + (self.endValue - self.startValue) * frac


def _raw(x):
    """Extrait le tenseur brut d'un objet `serie` (ou le renvoie tel quel si
    c'est déjà un tenseur / n'a pas d'attribut .data)."""
    return x.data if hasattr(x, "data") else x


class WeightedRegularizationTrainer:
    """
    Trainer standalone qui entraîne sur la TOTALITE des échantillons de deux
    bases (A = distribution cible, B = base de régularisation/perturbée),
    à chaque époque, mais pondère la contribution de chaque échantillon à la
    loss selon sa base d'origine :

      - poids constant `weightA` pour les échantillons de A (défaut 1.0)
      - poids `weightB(epoch)` pour les échantillons de B, qui décroît au fil
        des époques (annealing) pour que B agisse comme une régularisation
        de plus en plus légère, sans jamais dominer l'apprentissage de A.

    A chaque époque, TOUS les échantillons de A_train et TOUS ceux de
    B_train sont vus (aucun sous-échantillonnage), mélangés ensemble.
    """

    def __init__(
        self,
        weightA=1.0,
        weightB_start=1.0,
        weightB_end=0.05,
        weightMode="linear",
        weightDecayRate=0.95,
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
        self.weightA = weightA
        self.weightB_start = weightB_start
        self.weightB_end = weightB_end
        self.weightMode = weightMode
        self.weightDecayRate = weightDecayRate

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

        self.stats = SimpleNamespace(loss=[], valLossA=[], valLossB=[], lr=[], weightB=[])
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
    # Une époque d'entraînement (X, Y = `serie`, isADomain/baseWeights =
    # tenseurs bruts alignés avec X/Y, dans le même ordre). Le poids de B
    # est recalculé à CHAQUE STEP via self._globalStep, pas par époque.
    # ------------------------------------------------------------------
    def _training_phase(self, X, Y, model, optimizer, lrScheduler, isADomain, baseWeights):
        model.train()
        totalLoss = 0.0
        n = X.nSamples

        for start in tqdm(range(0, n, self.batchSize)):
            self.stats.lr.append(optimizer.param_groups[0]["lr"])

            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            isA_b = isADomain[start:start + self.batchSize]
            baseW_b = baseWeights[start:start + self.batchSize]

            # poids de B (et de A) évalué au step courant, pas à l'époque
            wB = self._scheduler.value(self._globalStep)
            self.stats.weightB.append(wB)

            domainWeight = torch.where(isA_b, torch.full_like(baseW_b, self.weightA), torch.full_like(baseW_b, wB))
            bW = (domainWeight * baseW_b).to(self.trainingDevice).to(self.trainingDtype)

            optimizer.zero_grad()
            preds = model(bX)

            raw_loss = nn.functional.mse_loss(_raw(preds), _raw(bY), reduction="none")
            # bW a la shape (batch,) -> on la broadcast sur les dims de features
            while bW.dim() < raw_loss.dim():
                bW = bW.unsqueeze(-1)
            loss = (raw_loss * bW).mean()

            loss.backward()
            optimizer.step()
            lrScheduler.step()
            self._globalStep += 1

            totalLoss += loss.item() * bX.nSamples

        totalLoss /= n
        return totalLoss

    # ------------------------------------------------------------------
    # Validation (non pondérée, pour suivre la vraie performance sur
    # chaque base séparément)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validation_phase(self, X, Y, model):
        model.eval()
        totalLoss = 0.0
        n = X.nSamples

        for start in range(0, n, self.batchSize):
            bX = X[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            bY = Y[start:start + self.batchSize].to(self.trainingDevice).to(self.trainingDtype)
            preds = model(bX)
            loss = nn.functional.mse_loss(_raw(preds), _raw(bY))
            totalLoss += loss.item() * bX.nSamples

        return totalLoss / n

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------
    def train(self, XA, YA, XB, YB, model, sampleWeightsA=None, sampleWeightsB=None):
        """
        sampleWeightsA / sampleWeightsB : poids *intra-base* optionnels
        (tenseur 1D de taille nSamples), s'ils existent déjà (ex: pour
        pondérer certains points particuliers au sein de A ou B).
        Ils sont multipliés par le poids de base (weightA / weightB(epoch)).
        """
        torch.cuda.empty_cache()
        gc.collect()

        XA_train, XA_val, YA_train, YA_val, _, _ = self._train_test_split(XA, YA)
        XB_train, XB_val, YB_train, YB_val, _, _ = self._train_test_split(XB, YB)

        nA, nB = XA_train.nSamples, XB_train.nSamples

        # concaténation une seule fois : la composition A/B ne change pas
        # d'une époque à l'autre, seul le poids appliqué à B évolue
        Xall = concat([XA_train, XB_train])
        Yall = concat([YA_train, YB_train])

        isA = torch.cat([torch.ones(nA, dtype=torch.bool), torch.zeros(nB, dtype=torch.bool)])

        baseWeights = torch.ones(nA + nB, dtype=torch.float32)
        if sampleWeightsA is not None:
            baseWeights[:nA] *= sampleWeightsA[:nA] if not hasattr(sampleWeightsA, "data") else _raw(sampleWeightsA[:nA]).squeeze()
        if sampleWeightsB is not None:
            baseWeights[nA:] *= sampleWeightsB[:nB] if not hasattr(sampleWeightsB, "data") else _raw(sampleWeightsB[:nB]).squeeze()

        model.to(self.trainingDevice).to(self.trainingDtype)
        optimizer, lrScheduler = self._build_optimizer(model)

        # nombre total de steps sur tout l'entraînement, pour caler le
        # scheduler de poids sur les steps plutôt que sur les époques
        stepsPerEpoch = math.ceil((nA + nB) / self.batchSize)
        totalSteps = self.epochs * stepsPerEpoch
        self._scheduler = WeightScheduler(
            self.weightB_start, self.weightB_end, totalSteps, self.weightMode, self.weightDecayRate
        )
        self._globalStep = 0

        for epoch in range(self.epochs):
            perm = torch.randperm(nA + nB)
            Xep, Yep = Xall[perm], Yall[perm]
            isAEp = isA[perm]
            baseWEp = baseWeights[perm]

            trainLoss = self._training_phase(Xep, Yep, model, optimizer, lrScheduler, isAEp, baseWEp)
            valLossA = self._validation_phase(XA_val, YA_val, model)
            valLossB = self._validation_phase(XB_val, YB_val, model)

            self.stats.loss.append(trainLoss)
            self.stats.valLossA.append(valLossA)
            self.stats.valLossB.append(valLossB)

            currentWB = self._scheduler.value(self._globalStep - 1)
            print(
                f"[{epoch + 1}/{self.epochs}] weightB (fin d'époque): {currentWB:.3f} | "
                f"Train: {self._format_loss(trainLoss)} | "
                f"Val A: {self._format_loss(valLossA)} | Val B: {self._format_loss(valLossB)} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        return model
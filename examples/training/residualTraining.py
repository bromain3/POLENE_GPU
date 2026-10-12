import POLENE_GPU.databaseGeneration.models.cantera0D as cantera0D
import POLENE_GPU.databaseGeneration.databaseGenerator as databaseGenerator
from POLENE_GPU.dataHandler.serie import concat, serie, load
from POLENE_GPU.preProcessing.scaler.stdScaler import stdScaler
from POLENE_GPU.preProcessing.scaler.BCT import boxCoxScaler
from POLENE_GPU.NN.models.resNet import ResNet
from POLENE_GPU.NN.trainer.baseTrainer import baseTrainer
from POLENE_GPU.NN.trainer.autoregressiveTrainer import autoregressiveTrainer
from POLENE_GPU.databaseGeneration.standalone.trajGeneration import generateTrajectoryTargets
from POLENE_GPU.NN.validator.oneStepPrediction import oneStepPredictionValidator
from POLENE_GPU.NN.validator.autoregressive0D import autoregressive0DValidator
from POLENE_GPU.NN.validator.residualsAnalysis import residualAnalysisValidator
from POLENE_GPU.misc.timer import Timer

import torch
import matplotlib.pyplot as plt

savingDevice = torch.device('cpu')
savingDdtype = torch.float64
CPUparallelProcessing = 30  # False or number of processor used
n_init_cond = 1000

trainingDevice = torch.device('cuda:0')
trainingDdtype = torch.float32

validationDevice = torch.device('cuda:0')
validationDdtype = torch.float64

JVPDtype = torch.float64
JVPDevice = torch.device('cuda:0')

mech = 'gri30.yaml'

with Timer("Model creation"):
    reactormodel = cantera0D.cstP(
        kinetics=mech,
        dtcfd=1e-6
    )

with Timer("DatabaseGenerator creation"):
    dtbgen = databaseGenerator.DatabaseGenerator(reactormodel, storingDevice=savingDevice, storingDtype=savingDdtype)



with Timer("Single trajectory"):
    X, Y, tX, tY = dtbgen.generateSingleTrajectory({'T': 1700, 'P': 101325, 'phi': 1.0})


with Timer("Initial conditions sampling"):
    intiCond = dtbgen.generateInitialConditionSamples('Halton', n_init_cond)

with Timer("Dataset generation"):
    #Xdtb, Ydtb = dtbgen.generateDataset(intiCond, CPUparallelProcessing=CPUparallelProcessing)

    #Ysdtb = generateTrajectoryTargets(
    #Xdtb,
    #seq_len=20,
    #dt=1e-6,
    #reactorType='cstP',
    #kinetics=mech,
    #n_workers=CPUparallelProcessing)


    #Xdtb.save("Xdtb.zarr", n_threads=CPUparallelProcessing)
    #for n, y in enumerate(Ysdtb):
    #    y.save(f"Ydtb_{n}.zarr", n_threads=CPUparallelProcessing)

    Xdtb = load("Xdtb.zarr", device=savingDevice, dtype=savingDdtype)
    Ysdtb = []
    # Xdtb = Xdtb.drop(osc_spe)
    for i in range(20):
        y = load(f"Ydtb_{i}.zarr", device=savingDevice, dtype=savingDdtype)#.drop(osc_spe)
        Ysdtb.append(y)

with Timer("Database scaling"):

    BCT4inc = boxCoxScaler(lambda_=0.1)
    stdScalerChem = stdScaler()
    scalerChem = BCT4inc + stdScalerChem

    # --- états (scaler unique pour tous les états chimiques) ---
    Xdtb_scaled = scalerChem.fit_transform(Xdtb)
    Ysdtb_scaled = [scalerChem.transform(y) for y in Ysdtb]   # cibles pour lossMode='state'

    # --- incréments dans l'espace BCT, un par pas de temps ---
    def _mk(d):
        return serie(d, colName=Xdtb.colName, storingDevice=savingDevice, storingDtype=savingDdtype)

    prev = BCT4inc.transform(Xdtb).data
    diffs = []
    for y in Ysdtb:
        cur = BCT4inc.transform(y).data
        diffs.append(cur - prev)      # BCT(y_k) - BCT(y_{k-1}), y_0 = Xdtb
        prev = cur

    # --- scaler unique pour tous les incréments : fit uniquement sur X -> Y[0] ---
    outScaler = stdScaler()
    outScaler.fit(_mk(diffs[0]))
    Ysdtb_inc_scaled = [outScaler.transform(_mk(d)) for d in diffs]  # cibles pour lossMode='increment'

with Timer("Model creation"):
    model = ResNet(Xdtb.nFeatures, 556, 4, 260, layerNorm=True, useBackbone=True, kaimigInit=False, activationOnOutput=False, predictState=False, chemScaler=BCT4inc, chemStdScaler=stdScalerChem)  # "larger" model, may be a bit more prone to overfitting with base training hyperparameters but has better pure mse precision
    #model = ResNet(Xdtb.nFeatures, 54, 2, 350, layerNorm=True, useBackbone=True, kaimigInit=False, activationOnOutput=False, predictState=False, chemScaler=BCT4inc, chemStdScaler=stdScalerChem)  # "small" network, usually more than enough

model.inputScaler = scalerChem
model.outputScaler = outScaler

with Timer("Trainer initialisation"):
    trainer = baseTrainer(trainingDtype=trainingDdtype, trainingDevice=trainingDevice, epochs=20, decaySteps=1500, lr=1e-3)  # for the smallest network

with Timer("Model training"):
# Trick to still be able to train the network directly on omega
    model.predictState = True
    trainer.train(Xdtb_scaled, Ysdtb_inc_scaled[0], model)
    model.predictState = False

with Timer("Validator initialisation"):
    validator = residualAnalysisValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validaton"):
    validator.validate(model, X, Y, plot=False)

#trainer.plotRanks()
#trainer.plotLoss()
#trainer.plotLr()

#with Timer("Validator initialisation"):
#    validator = oneStepPredictionValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

#with Timer("Model validaton"):
#    validator.validate(model, X, Y, tX)

#with Timer("Validator initialisation"):
#    validator = autoregressive0DValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

#with Timer("Model validaton"):
#    validator.validate(model, X, Y, tX, reactormodel.dtcfd)





with Timer("Trainer initialisation"):
    trainer = autoregressiveTrainer(trainingDtype=trainingDdtype, trainingDevice=trainingDevice, epochs=20, decaySteps=1500, lr=1e-4, warmupSteps=30000, lossMode='increment')  # for the smallest network

with Timer("Model training"):
    trainer.train(Xdtb_scaled, Ysdtb_scaled, model)


trainer.plotRanks()
trainer.plotLoss()
trainer.plotLr()

with Timer("Validator initialisation"):
    validator = residualAnalysisValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validaton"):
    validator.validate(model, X, Y, plot=False)

with Timer("Validator initialisation"):
    validator = oneStepPredictionValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validaton"):
    validator.validate(model, X, Y, tX)

with Timer("Validator initialisation"):
    validator = autoregressive0DValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validaton"):
    validator.validate(model, X, Y, tX, reactormodel.dtcfd)




import POLENE_GPU.databaseGeneration.models.cantera0D as cantera0D
import POLENE_GPU.databaseGeneration.databaseGenerator as databaseGenerator
from POLENE_GPU.preProcessing.scaler.stdScaler import stdScaler
from POLENE_GPU.preProcessing.scaler.logScaler import logScaler
from POLENE_GPU.NN.models.resNet import ResNet
from POLENE_GPU.NN.trainer.baseTrainer import baseTrainer
from POLENE_GPU.NN.validator.oneStepPrediction import oneStepPredictionValidator
from POLENE_GPU.NN.validator.autoregressive0D import autoregressive0DValidator
from POLENE_GPU.NN.validator.jacobianValidation import jacobianValidation
from POLENE_GPU.misc.timer import Timer

import kineticsGPU.torch.reactors.constantP as cstPreactorTorch

import torch
import matplotlib.pyplot as plt

savingDevice = torch.device('cpu')
savingDdtype = torch.float64
CPUparallelProcessing = 30  # False or number of processor used
n_init_cond = 1000

trainingDevice = torch.device('cuda:0')
trainingDdtype = torch.float32

validationDevice = torch.device('cuda:0')
validationDdtype = torch.float32

JVPDtype = torch.float64
JVPDevice = torch.device('cuda:0')

mech = 'gri30.yaml'

with Timer("Model creation"):
    reactormodel = cantera0D.cstP(
        kinetics=mech,
        cstSamplingdt=False,
    )

with Timer("DatabaseGenerator creation"):
    dtbgen = databaseGenerator.DatabaseGenerator(reactormodel, storingDevice=savingDevice, storingDtype=savingDdtype)

with Timer("Single trajectory"):
    X, Y, tX, tY = dtbgen.generateSingleTrajectory({'T': 1700, 'P': 101325, 'phi': 1.0})

with Timer("Initial conditions sampling"):
    intiCond = dtbgen.generateInitialConditionSamples('Halton', n_init_cond)

with Timer("Dataset generation"):
    Xdtb, Ydtb = dtbgen.generateDataset(intiCond, CPUparallelProcessing=CPUparallelProcessing)

with Timer("Model creation"):
    model = ResNet(Xdtb.nFeatures, 54, 2, 350, layerNorm=True)

with Timer("Database scaling"):
    scaler = stdScaler() # logScaler(threshold=1e-28) + stdScaler()
    Xdtb_scaled = scaler.fit_transform(Xdtb)
    Ydtb_scaled = scaler.transform(Ydtb)
    model.inputScaler = scaler
    model.outputScaler = scaler

with Timer("Trainer initialisation"):
    trainer = baseTrainer(trainingDtype=trainingDdtype, trainingDevice=trainingDevice, epochs=30)

with Timer("Model training"):
    trainer.train(Xdtb_scaled, Ydtb_scaled, model)

plt.semilogy(trainer.stats.loss, label='train')
plt.semilogy(trainer.stats.valLoss, label='val')
plt.legend()
plt.show()

plt.plot(trainer.stats.lr)
plt.show()

with Timer("Validator initialisation"):
    validator = oneStepPredictionValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validaton"):
    validator.validate(model, X, Y, tX)

with Timer("Validator initialisation"):
    validator = autoregressive0DValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validaton"):
    validator.validate(model, X, Y, tX, reactormodel.dtcfd)

with Timer("Validator initialisation"):
    validator = jacobianValidation(validationDtype=validationDdtype, validationDevice=validationDevice)
    reactorTorch = cstPreactorTorch.constantPReactorTPY(mech, JVPDevice, JVPDtype)

with Timer("Model validaton"):
    validator.validate(model, X, Y, tX, reactorTorch, reactormodel.dtcfd)


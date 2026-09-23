import POLENE_GPU.databaseGeneration.models.cantera0D as cantera0D
import POLENE_GPU.databaseGeneration.databaseGenerator as databaseGenerator
from POLENE_GPU.dataHandler.serie import concat
from POLENE_GPU.preProcessing.scaler.stdScaler import stdScaler
from POLENE_GPU.preProcessing.scaler.logScaler import logScaler
from POLENE_GPU.preProcessing.scaler.softLogScaler import softlogScaler
from POLENE_GPU.NN.models.resNet import ResNet
from POLENE_GPU.NN.models.adaptedResNet import ResNet as adaptedResNet
from POLENE_GPU.NN.models.integratorNet import IntegratorNet
from POLENE_GPU.NN.trainer.autoregressiveTrainer import autoregressiveTrainer
from POLENE_GPU.NN.validator.oneStepPrediction import oneStepPredictionValidator
from POLENE_GPU.NN.validator.autoregressive0D import autoregressive0DValidator
from POLENE_GPU.NN.validator.residualsAnalysis import residualAnalysisValidator
from POLENE_GPU.misc.timer import Timer
from POLENE_GPU.misc.JacobianComputation import getScaledTransformationJ, getScaledTransformationJExplicit, getChemicalJacobian, getTransformationJ
from POLENE_GPU.databaseGeneration.standalone.ILDMaugmentation import sampleManifoldAugmentation
from POLENE_GPU.databaseGeneration.standalone.trajGeneration import generateTrajectoryTargets
import POLENE_GPU.databaseGeneration.standalone.NeuralCFD as NeuNeu
from POLENE_GPU.exports.PYTHON import PYTHONexport


import kineticsGPU.torch.reactors.constantP as cstPreactorTorch

import torch
import matplotlib.pyplot as plt

savingDevice = torch.device('cpu')
savingDdtype = torch.float64
CPUparallelProcessing = 90  # False or number of processor used
n_init_cond = 1000

trainingDevice = torch.device('cuda:1')
trainingDdtype = torch.float32

validationDevice = torch.device('cuda:1')
validationDdtype = torch.float32

JVPDtype = torch.float64
JVPDevice = torch.device('cuda:1')

mech = 'gri30.yaml'

with Timer("Model creation"):
    reactormodel = cantera0D.cstP(
        kinetics=mech,
        dtcfd=1e-7
    )


with Timer("DatabaseGenerator creation"):
    dtbgen = databaseGenerator.DatabaseGenerator(reactormodel, storingDevice=savingDevice, storingDtype=savingDdtype)

with Timer("Single trajectory"):
    X, Y, tX, tY = dtbgen.generateSingleTrajectory({'T': 1700, 'P': 101325, 'phi': 1.0})

with Timer("Initial conditions sampling"):
    intiCond = dtbgen.generateInitialConditionSamples('Halton', n_init_cond)

with Timer("Dataset generation"):
    #Xdtb, Ydtb = dtbgen.generateDataset(intiCond, CPUparallelProcessing=CPUparallelProcessing)

    dtb_path = '../database_generation/NeuralCFD/sim_symetric_phi_1.0_T_2100_1500_cells_64bits_cantera_high_num_tol.h5'
    Xdtb, Ydtb = NeuNeu.readH5Database(dtb_path, mech)

    Ysdtb = generateTrajectoryTargets(
    Xdtb,
    seq_len=20,
    dt=1e-7,
    reactorType='cstV',
    kinetics=mech,
    n_workers=CPUparallelProcessing)

    osc_spe = ['NNH', 'CH2(S)', 'CH', 'CH2', 'CH2OH', 'HCNN', 'C', 'C2H', 'CN', 'HCO'] # HCO ?
    Xdtb = Xdtb.drop(osc_spe)
    Ydtb = Ydtb.drop(osc_spe)
    for n, y in enumerate(Ysdtb):
        Ysdtb[n] = y.drop(osc_spe)
    X = X.drop(osc_spe)
    Y = Y.drop(osc_spe)

with Timer("Model creation"):
    #model = ResNet(Xdtb.nFeatures, 54, 2, 350, layerNorm=True, useBackbone=True, kaimigInit=False, activationOnOutput=False)  # "small" network, usually more than enough
    model = ResNet(Xdtb.nFeatures, 556, 4, 260, layerNorm=True, useBackbone=True, kaimigInit=False, activationOnOutput=False)  # "small" network, usually more than enough

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


with Timer("Database scaling"):

    scaler = logScaler(threshold=1e-28) + stdScaler()
    Xdtb_scaled = scaler.fit_transform(Xdtb)
    Ydtb_scaled = scaler.transform(Ydtb)
    Ysdtb_scaled = []
    for y in Ysdtb:
        Ysdtb_scaled.append(scaler.transform(y))
    model.inputScaler = scaler
    model.outputScaler = scaler

with Timer("Trainer initialisation"):
    trainer = autoregressiveTrainer(trainingDtype=trainingDdtype, trainingDevice=trainingDevice, epochs=10, decaySteps=1500, lr=1e-3)  # for the smallest network

with Timer("Model training"):
    trainer.train(Xdtb_scaled, [Ydtb_scaled], model)

    export = PYTHONexport(model, Xdtb.colName, fictiveSpecies=osc_spe)
    export.save('export_PYTHON_pretrained.pln')

with Timer("Trainer initialisation"):
    trainer = autoregressiveTrainer(trainingDtype=trainingDdtype, trainingDevice=trainingDevice, epochs=20, decaySteps=1500, lr=1e-4, warmupSteps=30000)  # for the smallest network

with Timer("Model training"):
    trainer.train(Xdtb_scaled, Ysdtb_scaled, model)

    export = PYTHONexport(model, Xdtb.colName, fictiveSpecies=osc_spe)
    export.save('export_PYTHON_autoreg_trained.pln')

with Timer("Validator initialisation"):
    validator = oneStepPredictionValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validaton"):
    validator.validate(model, X, Y, tX)

with Timer("Validator initialisation"):
    validator = autoregressive0DValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validaton"):
    validator.validate(model, X, Y, tX, reactormodel.dtcfd)

with Timer("Validator initialisation"):
    validator = residualAnalysisValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validaton"):
    validator.validate(model, X, Y, plot=False)


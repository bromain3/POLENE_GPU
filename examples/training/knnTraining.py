import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # avant l'import de POLENE
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import POLENE_GPU.databaseGeneration.models.cantera0D as cantera0D
import POLENE_GPU.databaseGeneration.databaseGenerator as databaseGenerator
from POLENE_GPU.dataHandler.serie import concat
from POLENE_GPU.preProcessing.scaler.stdScaler import stdScaler
from POLENE_GPU.preProcessing.scaler.diffStdScaler import diffStdScaler
from POLENE_GPU.preProcessing.scaler.BCT import boxCoxScaler
from POLENE_GPU.preProcessing.scaler.logScaler import logScaler
from POLENE_GPU.preProcessing.scaler.softLogScaler import softlogScaler
from POLENE_GPU.NN.models.resNet import ResNet
from POLENE_GPU.NN.models.kNN import KNN
from POLENE_GPU.NN.models.adaptedResNet import ResNet as adaptedResNet
from POLENE_GPU.NN.models.MLP import MLP
from POLENE_GPU.NN.trainer.baseTrainer import baseTrainer
from POLENE_GPU.NN.validator.oneStepPrediction import oneStepPredictionValidator
from POLENE_GPU.NN.validator.autoregressive0D import autoregressive0DValidator
from POLENE_GPU.NN.validator.residualsAnalysis import residualAnalysisValidator
from POLENE_GPU.misc.timer import Timer
from POLENE_GPU.misc.JacobianComputation import getScaledTransformationJ, getScaledTransformationJExplicit, getChemicalJacobian, getTransformationJ

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
    Xdtb, Ydtb = dtbgen.generateDataset(intiCond, CPUparallelProcessing=CPUparallelProcessing)
    #Xdtb = Xdtb.drop(['NNH', 'CH2(S)', 'CH', 'CH2', 'CH2OH', 'HCNN', 'C', 'C2H', 'CN', 'HCO'])
    #Ydtb = Ydtb.drop(['NNH', 'CH2(S)', 'CH', 'CH2', 'CH2OH', 'HCNN', 'C', 'C2H', 'CN', 'HCO'])
    #X = X.drop(['NNH', 'CH2(S)', 'CH', 'CH2', 'CH2OH', 'HCNN', 'C', 'C2H', 'CN', 'HCO'])
    #Y = Y.drop(['NNH', 'CH2(S)', 'CH', 'CH2', 'CH2OH', 'HCNN', 'C', 'C2H', 'CN', 'HCO'])


with Timer("Model creation"):
    #model = ResNet(Xdtb.nFeatures, 100, 2, 350, layerNorm=True, useBackbone=True, kaimigInit=False, activationOnOutput=False)  #  "larger" model, may be a bit more prone to overfitting with base training hyperparameters but has better pure mse precision
    #model = ResNet(Xdtb.nFeatures, 54, 2, 350, layerNorm=True, useBackbone=True, kaimigInit=False, activationOnOutput=False)  # "small" network, usually more than enough
    model = KNN(n_neighbors=20)
    #model = adaptedResNet(Xdtb.nFeatures, 556, 4, 260)
    #model = IntegratorNet(
    #    input_dim=Xdtb.nFeatures,
    #    source_dim=55,  # taille de la représentation "terme source"
    #    source_layers=2,  # profondeur de la branche source
    #    source_width=350,  # largeur de la branche source
    #    integrator_layers=2,  # profondeur de l'intégrateur
    #    integrator_width=350,  # largeur de l'intégrateur
    #    layerNorm=True,
    #    activationOnOutput=False,
    #)
    #model = ResNet(Xdtb.nFeatures, 556, 4, 260, layerNorm=True, useBackbone=True, kaimigInit=False, activationOnOutput=False)  #  "larger" model, may be a bit more prone to overfitting with base training hyperparameters but has better pure mse precision
    #model = MLP(Xdtb.nFeatures, 6, 500, layerNorm=True)

with Timer("Database scaling"):

    #scaler = logScaler(threshold=1e-28)
    #scaler.NON_SPECIES_COLS = []
    #scaler = scaler + stdScaler()

    #scaler = logScaler(threshold=1e-28) + stdScaler() #ShiftedLogScaler(min_shift=1e-10) + stdScaler()
    a = stdScaler()
    scaler = boxCoxScaler(lambda_=0.1, threshold=0.0) + a
    #scaler = stdScaler()
    #scaler = softlogScaler(threshold=1e-28, verbose=True) + stdScaler()
    scaler = scaler.fit(Xdtb)
    Xdtb_scaled = scaler.transform(Xdtb)
    Ydtb_scaled = scaler.transform(Ydtb)
    model.inputScaler = scaler
    model.outputScaler = scaler
    print(a.std)

with Timer("Model training"):
    model.fit(Xdtb_scaled, Ydtb_scaled)

with Timer("Validator initialisation"):
    validator = oneStepPredictionValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validaton"):
    validator.validate(model, X, Y, tX)

with Timer("Validator initialisation"):
    validator = autoregressive0DValidator(validationDtype=validationDdtype, validationDevice=validationDevice)
    #TODO: calculer la correlation entre N2 et T et montrer qu'elle va dans le bon sens et que sa valeur est liée à la physique
with Timer("Model validaton"):
    validator.validate(model, X, Y, tX, reactormodel.dtcfd)

with Timer("Validator initialisation"):
    validator = residualAnalysisValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validaton"):
    validator.validate(model, X, Y, plot=False)


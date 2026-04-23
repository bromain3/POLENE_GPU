import POLENE_GPU.databaseGeneration.models.cantera0D as cantera0D
import POLENE_GPU.databaseGeneration.databaseGenerator as databaseGenerator
from POLENE_GPU.preProcessing.scaler.stdScaler import stdScaler
from POLENE_GPU.preProcessing.scaler.logScaler import logScaler
from POLENE_GPU.NN.models.resNet import ResNet
from POLENE_GPU.databaseGeneration.standalone.oversampler import generateIntermediateSamples
from POLENE_GPU.NN.trainer.baseTrainer import baseTrainer
from POLENE_GPU.NN.trainer.mahanalobisTrainer import mahanalobisTrainer, jacobianMahanalobisTrainer, mahanalobisTrainerGradNorm
from POLENE_GPU.NN.validator.oneStepPrediction import oneStepPredictionValidator
from POLENE_GPU.NN.validator.autoregressive0D import autoregressive0DValidator
from POLENE_GPU.NN.validator.jacobianValidation import jacobianValidation
from POLENE_GPU.misc.timer import Timer
from POLENE_GPU.misc.batchFunction import batched_apply
from POLENE_GPU.dataHandler.serie import serie
import POLENE_GPU.databaseGeneration.standalone.NeuralCFD as NeuNeu
from POLENE_GPU.misc.JacobianComputation import getScaledTransformationJ_1


import kineticsGPU.torch.reactors.constantV as cstVreactorTorch
import kineticsGPU.cuda.kernels.per_thread.reactor_odes as reactors

import torch
import matplotlib.pyplot as plt

savingDevice = torch.device('cpu')
savingDdtype = torch.float64
CPUparallelProcessing = 30  # False or number of processor used
n_init_cond = 1000

trainingDevice = torch.device('cuda:1')
trainingDdtype = torch.float32

validationDevice = torch.device('cuda:1')
validationDdtype = torch.float32

JVPDtype = torch.float64
JVPDevice = torch.device('cuda:1')

mech = 'gri30.yaml'
dtb_path = '../database_generation/NeuralCFD/sim_symetric_phi_1.0_T_2100_1500_cells.h5'
undersamplingProp = 0.3  # TODO: study the impact of oversampling before or after the uniform undersampling. I think the cleanest would be to oversample and then undersample
classicTraining = True  #  ~ 1.731e-5 en val avec training classique (test == val étonnement)
dtcfd = 1e-7

with Timer("Model creation"):
    reactormodel = cantera0D.cstV(
        kinetics=mech,
        cstSamplingdt=False,
        dtcfd=dtcfd,
    )

with Timer("DatabaseGenerator creation"):
    dtbgen = databaseGenerator.DatabaseGenerator(reactormodel, storingDevice=savingDevice, storingDtype=savingDdtype)

with Timer("Single trajectory"):
    X, Y, tX, tY = dtbgen.generateSingleTrajectory({'T': 1700, 'P': 101325, 'phi': 1.0})

#with Timer("Initial conditions sampling"):
#    intiCond = dtbgen.generateInitialConditionSamples('Halton', n_init_cond)

#with Timer("Dataset generation"):
#    Xdtb, Ydtb = dtbgen.generateDataset(intiCond, CPUparallelProcessing=CPUparallelProcessing)

with Timer("Dataset generation"):
    Xdtb, Ydtb = NeuNeu.readH5Database(dtb_path, mech)
    n = int(Xdtb.nSamples * undersamplingProp)
    indices = torch.randperm(Xdtb.nSamples)[:n]
    Xdtb = Xdtb[indices]
    Ydtb = Ydtb[indices]

with Timer("Oversampling"):
    X_more, Y_more = generateIntermediateSamples(Xdtb, dtcfd, "cstV", mech, CPUparallelProcessing, chunksize=1500)
    # TODO: il faudrait soint concat X_more et Y_more à Xdtb et Ydtb soit que X_more et Y_more incorporent directement X_dtb et Y_dtb mais pour l'instant on laisse tomber
    n = int(X_more.nSamples * undersamplingProp)
    indices = torch.randperm(X_more.nSamples)[:n]
    X_more = X_more[indices]
    Y_more = Y_more[indices]

with Timer("Model creation"):
    model = ResNet(X_more.nFeatures, 54, 2, 350, layerNorm=True)

with Timer("Database scaling"):
    scaler = stdScaler()
    Xdtb_scaled = scaler.fit_transform(X_more)
    Ydtb_scaled = scaler.transform(Y_more)
    model.inputScaler = scaler
    model.outputScaler = scaler

if not classicTraining:
    with Timer("Computing per thread jacobian"):
        J_Y_scaled = getScaledTransformationJ_1("cstV", mech, Xdtb_scaled, Ydtb_scaled, model.inputScaler, model.outputScaler,
                                                reactormodel.dtcfd, computationArchitechture='pt', batch_size=15_000,
                                                computationDevice=JVPDevice, storingDevice=savingDevice,
                                                storingDtype=savingDdtype)
        print(J_Y_scaled.data.max())

with Timer("Trainer initialisation"):
    if not classicTraining:
        trainer = mahanalobisTrainer(trainingDtype=trainingDdtype, trainingDevice=trainingDevice, epochs=30, lamb=3)
        #trainer = mahanalobisTrainerGradNorm(trainingDtype=trainingDdtype, trainingDevice=trainingDevice, epochs=30)
    else:
        trainer = baseTrainer(trainingDtype=trainingDdtype, trainingDevice=trainingDevice, epochs=30)

with Timer("Model training"):
    if not classicTraining:
        trainer.train(Xdtb_scaled, Ydtb_scaled, J_Y_scaled, model)
    else:
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
    reactorTorch = cstVreactorTorch.constantVReactorTPY(mech, JVPDevice, JVPDtype)

with Timer("Model validaton"):
    validator.validate(model, X, Y, tX, reactorTorch, reactormodel.dtcfd)


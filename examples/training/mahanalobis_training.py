import POLENE_GPU.databaseGeneration.models.cantera0D as cantera0D
import POLENE_GPU.databaseGeneration.databaseGenerator as databaseGenerator
from POLENE_GPU.preProcessing.scaler.stdScaler import stdScaler
from POLENE_GPU.preProcessing.scaler.logScaler import logScaler
from POLENE_GPU.NN.models.resNet import ResNet
from POLENE_GPU.NN.trainer.baseTrainer import baseTrainer
from POLENE_GPU.NN.trainer.mahanalobisTrainer import mahanalobisTrainer, jacobianMahanalobisTrainer, mahanalobisTrainerGradNorm
from POLENE_GPU.NN.validator.oneStepPrediction import oneStepPredictionValidator
from POLENE_GPU.NN.validator.autoregressive0D import autoregressive0DValidator
from POLENE_GPU.NN.validator.jacobianValidation import jacobianValidation
from POLENE_GPU.misc.timer import Timer
from POLENE_GPU.misc.batchFunction import batched_apply
from POLENE_GPU.dataHandler.serie import serie
import POLENE_GPU.databaseGeneration.standalone.NeuralCFD as NeuNeu

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
dtb_path = '../database_generation/NeuralCFD/sim_symetric_phi_1.0_T_2100.h5'
undersamplingProp = 0.3
classicTraining = False  #  ~ 1.731e-5 en val avec training classique (test == val étonnement)


with Timer("Model creation"):
    reactormodel = cantera0D.cstV(
        kinetics=mech,
        cstSamplingdt=False,
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

with Timer("Model creation"):
    model = ResNet(Xdtb.nFeatures, 54, 2, 350, layerNorm=True)

with Timer("Database scaling"):
    scaler = stdScaler()
    Xdtb_scaled = scaler.fit_transform(Xdtb)
    Ydtb_scaled = scaler.transform(Ydtb)
    model.inputScaler = scaler
    model.outputScaler = scaler

if not classicTraining:
    with Timer("Jacobian computator initialisation"):
        jacobianComp = reactors.CUDAFullJacobianConstantVTPY_PT(mech, torchComputationDtype=JVPDtype, dataDevice=savingDevice, computationDevice=JVPDevice, storageDtype=savingDdtype)

    with Timer("Computing per thread jacobian"):
        J_chem_Y = batched_apply(jacobianComp.evaluate, Ydtb.data[:, 0].flatten(), Ydtb.data[:, 1].flatten(), Ydtb.data[:, 2:], batch_size=15_000, device=JVPDevice, dtype=JVPDtype)
        J_chem_Y = serie(J_chem_Y, ['J'], storingDevice=savingDevice, storingDtype=savingDdtype)

        #TODO: faire passer ces opérations sur GPU en batch apply (on pourrait aussi tout faire en torch32, à voir si ça modifie beacoup la precision):
        #On calcule -dt * J_chem
        J_transfo_Y_data = - reactormodel.dtcfd * J_chem_Y.data
        # On ajoute 1 sur la diagonale de chaque matrice du batch (Broadcasting efficace) plutot que de calculer/stocker une grande quantité de matrices identité
        n = Xdtb.nFeatures
        J_transfo_Y_data[:, range(n), range(n)] += 1
        J_transfo_Y = serie(J_transfo_Y_data, ['J'], storingDevice=savingDevice, storingDtype=savingDdtype)
        del J_chem_Y
        J_transfo_Y_scaled = model.outputScaler.J(Ydtb).unsqueeze(2) * J_transfo_Y.data * model.inputScaler.J_1(Ydtb_scaled).unsqueeze(1)  #TODO: verifier ce calcul. si pas sur, transformer tout en matrices diagonales, tant pis pour le temps de calcul

        J_transfo_Y_scaled = serie(J_transfo_Y_scaled, ['J'], storingDevice=savingDevice, storingDtype=savingDdtype)
        del J_transfo_Y



with Timer("Trainer initialisation"):
    if not classicTraining:
        trainer = mahanalobisTrainer(trainingDtype=trainingDdtype, trainingDevice=trainingDevice, epochs=30, lamb=3)
        #trainer = mahanalobisTrainerGradNorm(trainingDtype=trainingDdtype, trainingDevice=trainingDevice, epochs=30)
    else:
        trainer = baseTrainer(trainingDtype=trainingDdtype, trainingDevice=trainingDevice, epochs=30)

with Timer("Model training"):
    if not classicTraining:
        trainer.train(Xdtb_scaled, Ydtb_scaled, J_transfo_Y_scaled, model)
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


import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # avant l'import de POLENE
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import POLENE_GPU.databaseGeneration.models.cantera0D as cantera0D
import POLENE_GPU.databaseGeneration.databaseGenerator as databaseGenerator
from POLENE_GPU.dataHandler.serie import serie, load
from POLENE_GPU.preProcessing.scaler.stdScaler import stdScaler
from POLENE_GPU.preProcessing.scaler.BCT import boxCoxScaler
from POLENE_GPU.NN.models.kNN import KNN

from POLENE_GPU.NN.trainer.omegaTrainer import omegaTrainer
from POLENE_GPU.NN.trainer.baseTrainer import baseTrainer
from POLENE_GPU.NN.validator.oneStepPrediction import oneStepPredictionValidator
from POLENE_GPU.NN.validator.autoregressive0D import autoregressive0DValidator
from POLENE_GPU.NN.validator.residualsAnalysis import residualAnalysisValidator
from POLENE_GPU.misc.timer import Timer

import torch

savingDevice = torch.device('cpu')
savingDdtype = torch.float64
CPUparallelProcessing = 30
n_init_cond = 1000

trainingDevice = torch.device('cuda:0')
trainingDdtype = torch.float32

validationDevice = torch.device('cuda:0')
validationDdtype = torch.float64

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


with Timer("Database scaling"):
    BCT4inc = boxCoxScaler(lambda_=0.1, threshold=0.0)
    stdScalerChem = stdScaler()
    scalerChem = BCT4inc + stdScalerChem

    # 1. Normalisation des états TPY
    Xdtb_scaled = scalerChem.fit_transform(Xdtb)
    Ydtb_scaled = scalerChem.transform(Ydtb)

    # 2. Précalcul haute précision des incréments en espace BCT
    def _mk(d):
        return serie(d, colName=Xdtb.colName, storingDevice=savingDevice, storingDtype=savingDdtype)

    prev = BCT4inc.transform(Xdtb).data
    cur = BCT4inc.transform(Ydtb).data
    diff = cur - prev

    # 3. Normalisation de l'incrément oméga cible
    outScaler = stdScaler()
    inc_dtb_scaled = outScaler.fit_transform(_mk(diff))
#TODO: refaire l'api parceque la pour l'instant ça commence à partir en couilles
with Timer("Model creation"):
    model = KNN(
        chemScaler=BCT4inc, chemStdScaler=stdScalerChem, predictState=False
    )

model.inputScaler = scalerChem
model.outputScaler = scalerChem

model.fit(Xdtb_scaled, inc_dtb_scaled)


with Timer("Validator initialisation"):
    validator = residualAnalysisValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validation"):
    validator.validate(model, X, Y, plot=False)

with Timer("Validator initialisation (One step)"):
    oneStepVal = oneStepPredictionValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validation (One step)"):
    oneStepVal.validate(model, X, Y, tX)

with Timer("Validator initialisation (Autoregressive)"):
    autoRegVal = autoregressive0DValidator(validationDtype=validationDdtype, validationDevice=validationDevice)

with Timer("Model validation (Autoregressive)"):
    autoRegVal.validate(model, X, Y, tX, reactormodel.dtcfd)
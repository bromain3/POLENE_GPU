import POLENE_GPU.databaseGeneration.models.cantera0D as cantera0D
import POLENE_GPU.databaseGeneration.databaseGenerator as databaseGenerator
from POLENE_GPU.preProcessing.scaler.stdScaler import stdScaler
from POLENE_GPU.preProcessing.scaler.logScaler import logScaler
from POLENE_GPU.preProcessing.scaler.shiftedLogScaler import ShiftedLogScaler
from POLENE_GPU.NN.models.resNet import ResNet
from POLENE_GPU.NN.models.adaptedResNet import ResNet as adaptedResNet
from POLENE_GPU.misc.MixtureOfExpertsPreparation import prepare_moe_datasets_shared_scaler, prepare_moe_datasets_per_cluster_scaler
from POLENE_GPU.NN.models.mixtureOfExperts import MixtureOfExperts
from POLENE_GPU.NN.trainer.baseTrainer import baseTrainer
from POLENE_GPU.NN.validator.oneStepPrediction import oneStepPredictionValidator
from POLENE_GPU.NN.validator.autoregressive0D import autoregressive0DValidator
from POLENE_GPU.NN.validator.residualsAnalysis import residualAnalysisValidator
from POLENE_GPU.misc.timer import Timer

import torch
import matplotlib.pyplot as plt

# ── devices / dtypes ──────────────────────────────────────────────────────────
savingDevice          = torch.device('cpu')
savingDdtype          = torch.float64
CPUparallelProcessing = 30
n_init_cond           = 1000

trainingDevice = torch.device('cuda:0')
trainingDdtype = torch.float32

validationDevice = torch.device('cuda:0')
validationDdtype = torch.float32

# ── MoE config ────────────────────────────────────────────────────────────────
n_clusters = 4   # number of experts

mech = 'gri30.yaml'

# ── reactor & data generation  (unchanged) ───────────────────────────────────
with Timer("Model creation"):
    reactormodel = cantera0D.cstP(
        kinetics=mech,
    )

with Timer("DatabaseGenerator creation"):
    dtbgen = databaseGenerator.DatabaseGenerator(
        reactormodel, storingDevice=savingDevice, storingDtype=savingDdtype
    )

with Timer("Single trajectory"):
    X, Y, tX, tY = dtbgen.generateSingleTrajectory({'T': 1700, 'P': 101325, 'phi': 1.0})

with Timer("Initial conditions sampling"):
    intiCond = dtbgen.generateInitialConditionSamples('Halton', n_init_cond)

with Timer("Dataset generation"):
    Xdtb, Ydtb = dtbgen.generateDataset(intiCond, CPUparallelProcessing=CPUparallelProcessing)

# ── MoE dataset split ─────────────────────────────────────────────────────────
with Timer("MoE dataset split"):
    def sc_class():
        return logScaler(threshold=1e-28) + stdScaler()

    clusterer, clustering_scaler, X_list, Y_list, scaler_list = prepare_moe_datasets_per_cluster_scaler(  #TODO: it seems that there is an error at least in the cluster colouring in validation but maybe also in the clustering whene the clustering scaler is not the same type as the models scalers
        Xdtb, Ydtb,
        clustering_scaler=logScaler(threshold=1e-28) + stdScaler(),
        expert_scaler_cls=sc_class,
        n_clusters=n_clusters,
    )
    print(f"  Cluster sizes: {[x.nSamples for x in X_list]}")

# ── train one expert per cluster ──────────────────────────────────────────────
experts = []

for k, (X_sub, Y_sub, scaler) in enumerate(zip(X_list, Y_list, scaler_list)):
    print(f"\n{'='*60}")
    print(f"  Expert {k+1}/{n_clusters}  —  {X_sub.nSamples} samples")
    print(f"{'='*60}")

    with Timer(f"  [{k}] Database scaling"):
        X_scaled = scaler.transform(X_sub)
        Y_scaled = scaler.transform(Y_sub)

    with Timer(f"  [{k}] Model creation"):
        model = ResNet(
            X_sub.nFeatures, 256, 4, 500,
            layerNorm=True, useBackbone=True,
            kaimigInit=False, activationOnOutput=False,
        )
        model.inputScaler  = scaler
        model.outputScaler = scaler

    with Timer(f"  [{k}] Trainer initialisation"):
        trainer = baseTrainer(
            trainingDtype=trainingDdtype,
            trainingDevice=trainingDevice,
            epochs=50,
            decaySteps=850,
            lr=1e-3
        )

    with Timer(f"  [{k}] Model training"):
        trainer.train(X_scaled, Y_scaled, model)

    experts.append(model)

# ── assemble MoE wrapper ──────────────────────────────────────────────────────
with Timer("MoE assembly"):
    moe = MixtureOfExperts(experts, clusterer, clustering_scaler)
    print(moe)

# ── validation (same validators as before, now on the MoE model) ──────────────
with Timer("Validator initialisation"):
    validator = oneStepPredictionValidator(
        validationDtype=validationDdtype, validationDevice=validationDevice
    )

with Timer("Model validation"):
    validator.validate(moe, X, Y, tX, showClusters=True)

with Timer("Validator initialisation"):
    validator = autoregressive0DValidator(
        validationDtype=validationDdtype, validationDevice=validationDevice
    )

with Timer("Model validation"):
    validator.validate(moe, X, Y, tX, reactormodel.dtcfd, showClusters=True)

with Timer("Validator initialisation"):
    validator = residualAnalysisValidator(
        validationDtype=validationDdtype, validationDevice=validationDevice
    )

with Timer("Model validation"):
    validator.validate(moe, X, Y, plot=True)

plt.show()
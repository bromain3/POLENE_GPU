import POLENE_GPU.databaseGeneration.models.cantera0D as cantera0D
import POLENE_GPU.databaseGeneration.databaseGenerator as databaseGenerator
from POLENE_GPU.preProcessing.scaler.stdScaler import stdScaler
from POLENE_GPU.preProcessing.scaler.logScaler import logScaler
from POLENE_GPU.preProcessing.scaler.shiftedLogScaler import ShiftedLogScaler
from POLENE_GPU.NN.models.resNet import ResNet
from POLENE_GPU.NN.models.integratorNet import IntegratorNet
from POLENE_GPU.NN.trainer.baseTrainer import baseTrainer
from POLENE_GPU.NN.validator.oneStepPrediction import oneStepPredictionValidator
from POLENE_GPU.NN.validator.autoregressive0D import autoregressive0DValidator
from POLENE_GPU.NN.validator.jacobianValidation import jacobianValidation
from POLENE_GPU.misc.timer import Timer

import kineticsGPU.torch.reactors.constantP as cstPreactorTorch

import torch
import matplotlib.pyplot as plt
import optuna
from optuna.pruners import MedianPruner

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


with Timer("Database scaling"):
    scaler = stdScaler() #ShiftedLogScaler(min_shift=1e-10) + stdScaler()
    Xdtb_scaled = scaler.fit_transform(Xdtb)
    Ydtb_scaled = scaler.transform(Ydtb)


# ── Fonction objectif ─────────────────────────────────────────────────────────
def objective(trial):

    # --- Hyperparamètres du modèle ---
    source_dim       = trial.suggest_int("source_dim", 32, 512, step=20)
    source_layers    = trial.suggest_int("source_layers", 1, 4)
    source_width     = trial.suggest_int("source_width", 32, 512, step=20)
    integrator_layers = trial.suggest_int("integrator_layers", 1, 4)
    integrator_width  = trial.suggest_int("integrator_width", 32, 512, step=20)

    # --- Hyperparamètres du trainer ---
    #epochs      = trial.suggest_int("epochs", 20, 60, step=10)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    decaySteps = trial.suggest_int("decay_steps", 500, 10000, step=10)
    decayRate = trial.suggest_float("decay_rate", 0.6, 1.0)
    # Si baseTrainer expose lr / batch_size / decay_steps, les ajouter ici :
    # lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # decay_steps = trial.suggest_int("decay_steps", 500, 3000, step=500)

    # --- Construction du modèle ---
    model = IntegratorNet(
        input_dim=Xdtb.nFeatures,
        source_dim=source_dim,
        source_layers=source_layers,
        source_width=source_width,
        integrator_layers=integrator_layers,
        integrator_width=integrator_width,
        layerNorm=True,
        activationOnOutput=False,
    )
    model.inputScaler  = scaler   # scaler déjà fitté, on le réutilise
    model.outputScaler = scaler

    # --- Trainer ---
    trainer = baseTrainer(
        trainingDtype=trainingDdtype,
        trainingDevice=trainingDevice,
        epochs=30,
        lr=lr,
        decaySteps=decaySteps,
        decayRate=decayRate,
    )


    trainer.train(Xdtb_scaled, Ydtb_scaled, model)

    # --- Métrique à minimiser ---
    val_loss = trainer.stats.valLoss[-1]
    return val_loss


# ── Lancement de l'étude ──────────────────────────────────────────────────────
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(),   # bayésien (TPE)
    study_name="test",
    storage="sqlite:///test.db",   # persistance → relance possible
    load_if_exists=True,
)

study.optimize(
    objective,
    #n_trials=50,
    #timeout=3600 * 4,      # 4h max
    n_jobs=1,              # 1 GPU → séquentiel
    show_progress_bar=True,
)

# ── Résultats ─────────────────────────────────────────────────────────────────
print("Best valLoss :", study.best_value)
print("Best params  :", study.best_params)

# Visualisations Optuna
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
optuna.visualization.plot_contour(study).show()
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
import POLENE_GPU.databaseGeneration.standalone.NeuralCFD as NeuNeu

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
dtb_path = '../database_generation/NeuralCFD/sim_symetric_phi_1.0_T_2100_1500_cells.h5'
undersamplingProp = 0.3
dtcfd = 1e-7

with Timer("Dataset generation"):
    Xdtb, Ydtb = NeuNeu.readH5Database(dtb_path, mech)
    n = int(Xdtb.nSamples * undersamplingProp)
    indices = torch.randperm(Xdtb.nSamples)[:n]
    Xdtb = Xdtb[indices]
    Ydtb = Ydtb[indices]


with Timer("Database scaling"):
    scaler = stdScaler() #ShiftedLogScaler(min_shift=1e-10) + stdScaler()
    Xdtb_scaled = scaler.fit_transform(Xdtb)
    Ydtb_scaled = scaler.transform(Ydtb)


# ── Fonction objectif ─────────────────────────────────────────────────────────
def objective(trial):

    # --- Hyperparamètres du modèle ---
    #backbone_dim       = trial.suggest_int("backbone_dim", 16, 512, step=20)
    #nb_blocks    = trial.suggest_int("nb_blocks", 1, 10)
    #hidden_dim     = trial.suggest_int("hidden_dim", 16, 1024, step=20)


    # --- Hyperparamètres du trainer ---
    #epochs      = trial.suggest_int("epochs", 20, 60, step=10)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    decaySteps = trial.suggest_int("decay_steps", 500, 10000, step=10)
    decayRate = trial.suggest_float("decay_rate", 0.6, 1.0)
    beta1 = trial.suggest_float("beta1", 0.7, 1.0, log=False)
    oneminusbeta2 = trial.suggest_float("1mbeta2", 1e-6, 1e-1, log=True)
    beta2 = 1 - oneminusbeta2

    # Si baseTrainer expose lr / batch_size / decay_steps, les ajouter ici :
    # lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # decay_steps = trial.suggest_int("decay_steps", 500, 3000, step=500)

    # --- Construction du modèle ---
    model = ResNet(Xdtb.nFeatures, 54, 2, 350, layerNorm=True, useBackbone=True, kaimigInit=False, activationOnOutput=False)  #  "larger" model, may be a bit more prone to overfitting with base training hyperparameters but has better pure mse precision

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
        beta1=beta1,
        beta2=beta2
    )


    trainer.train(Xdtb_scaled, Ydtb_scaled, model)

    # --- Métrique à minimiser ---
    val_loss = trainer.stats.valLoss[-1]
    return val_loss


# ── Lancement de l'étude ──────────────────────────────────────────────────────
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(),   # bayésien (TPE)
    study_name="betas_opt",
    storage="sqlite:///betas_opt.db",   # persistance → relance possible
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
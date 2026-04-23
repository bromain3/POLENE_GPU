import cantera as ct
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

from POLENE_GPU.dataHandler.serie import serie

# --- State global par worker (initialisé une seule fois) ---
_worker_gas_feat = None
_worker_gas_targ = None
_worker_reactor_class = None
_worker_dt = None

def _worker_init(kinetics, reactorType, dt):
    """Appelé une seule fois à la création du worker — parse le YAML ici."""
    global _worker_gas_feat, _worker_gas_targ, _worker_reactor_class, _worker_dt

    if reactorType == 'cstV':
        _worker_reactor_class = ct.IdealGasReactor
    elif reactorType == 'cstP':
        _worker_reactor_class = ct.IdealGasConstPressureReactor
    else:
        raise ValueError(f"reactorType must be 'cstV' or 'cstP', got '{reactorType}'")

    _worker_gas_feat = ct.Solution(kinetics)
    _worker_gas_targ = ct.Solution(kinetics)
    _worker_dt = dt


def _worker_task(state):
    """Tâche par point — réutilise le gas déjà chargé."""
    global _worker_gas_feat, _worker_gas_targ, _worker_reactor_class, _worker_dt

    X_new = []
    Y_new = []
    dt = _worker_dt

    T_ini = float(state[0])
    P_ini = float(state[1])
    Y_ini = state[2:]

    _worker_gas_feat.TPY = T_ini, P_ini, Y_ini
    reactor_feat = _worker_reactor_class(_worker_gas_feat)
    net_feat = ct.ReactorNet([reactor_feat])

    _worker_gas_targ.TPY = T_ini, P_ini, Y_ini
    reactor_targ = _worker_reactor_class(_worker_gas_targ)
    net_targ = ct.ReactorNet([reactor_targ])

    net_feat.step()

    # TODO: étudier cette partie car le premiier pas de temps est fait sur la base d'une formule différented des autres.
    #  On pourait soit trouver un lien entre pas de temps adaptatif et nombre de conditionement pour faire les pas de temps "à la main"
    #  soit on lance une première intégration entre t et dt, on calcul le nombre de pas de temps intermédiaires ou le pas de temps moyen
    #  et on relance l'intégration avec comme pas de temps initiale le pas de temps calculé.
    #  cela permettrait probablement également de pouvoir intégrer avec le solveur cuda et donc de gagner du temps
    if net_feat.time >= dt:
        return X_new, Y_new

    while net_feat.time < dt:
        x_state = np.array(
            [reactor_feat.T, reactor_feat.thermo.P] + list(reactor_feat.thermo.Y)
        )
        net_targ.advance(net_feat.time + dt)
        y_state = np.array(
            [reactor_targ.T, reactor_targ.thermo.P] + list(reactor_targ.thermo.Y)
        )
        X_new.append(x_state)
        Y_new.append(y_state)
        net_feat.step()

    return X_new, Y_new


def generateIntermediateSamples(X, dt, reactorType, kinetics, n_workers=None, chunksize=50):
    col_names = X.colName
    states = X.data.cpu().detach().numpy()  # shape (N, nCols)

    if n_workers is None:
        n_workers = mp.cpu_count()

    X_new_list = []
    Y_new_list = []

    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(kinetics, reactorType, dt)
    ) as pool:
        results = list(tqdm(
            pool.imap(_worker_task, states, chunksize=chunksize),
            total=len(states)
        ))

    for X_new, Y_new in results:
        X_new_list += X_new
        Y_new_list += Y_new

    if len(X_new_list) == 0:
        raise ValueError("No new sample generated")

    X_arr = np.array(X_new_list)
    Y_arr = np.array(Y_new_list)

    X_out = serie(X_arr, col_names, storingDevice=X.storingDevice, storingDtype=X.storingDtype)
    Y_out = serie(Y_arr, col_names, storingDevice=X.storingDevice, storingDtype=X.storingDtype)

    return X_out, Y_out




def generateIntermediateSamplesBatched(X, dt, reactorType, kinetics, n_workers=None, chunksize=50, accumulation_batch=100_000):
    col_names = X.colName
    states = X.data.cpu().detach().numpy()

    if n_workers is None:
        n_workers = mp.cpu_count()

    n_cols = states.shape[1]

    # Buffers courants — on flush dans chunks_X/Y quand ils atteignent accumulation_batch
    buffer_X = []
    buffer_Y = []
    buffer_size = 0

    # Liste de blocs numpy — un np.concatenate final, pas N
    chunks_X = []
    chunks_Y = []
    total_samples = 0

    def _flush_buffer():
        nonlocal buffer_X, buffer_Y, buffer_size
        if buffer_size == 0:
            return
        chunks_X.append(np.array(buffer_X, dtype=np.float64))
        chunks_Y.append(np.array(buffer_Y, dtype=np.float64))
        buffer_X = []
        buffer_Y = []
        buffer_size = 0

    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(kinetics, reactorType, dt)
    ) as pool:
        with tqdm(total=len(states), desc="Augmentation") as pbar:
            for X_new, Y_new in pool.imap(_worker_task, states, chunksize=chunksize):
                n = len(X_new)
                if n > 0:
                    buffer_X += X_new
                    buffer_Y += Y_new
                    buffer_size += n
                    total_samples += n

                    # Flush si le buffer est assez grand
                    if buffer_size >= accumulation_batch:
                        _flush_buffer()

                pbar.update(1)
                pbar.set_postfix({"nouveaux samples": total_samples})

        # Flush du reste
        _flush_buffer()

    if total_samples == 0:
        raise ValueError("No new sample generated")

    # Un seul concatenate final
    X_arr = np.concatenate(chunks_X, axis=0)
    Y_arr = np.concatenate(chunks_Y, axis=0)

    X_out = serie(X_arr, col_names, storingDevice=X.storingDevice, storingDtype=X.storingDtype)
    Y_out = serie(Y_arr, col_names, storingDevice=X.storingDevice, storingDtype=X.storingDtype)

    return X_out, Y_out
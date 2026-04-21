from POLENE_GPU.dataHandler.serie import serie
from POLENE_GPU.misc.batchFunction import batched_apply

import kineticsGPU.cuda.kernels.per_thread.reactor_odes as reactors_pt
import kineticsGPU.cuda.kernels.per_block.reactor_odes_pb as reactors_pb

import torch

def getChemicalJacobian(reactorType, mech, X, computationArchitechture='pt', batch_size=15_000, computationDevice=None, storingDevice=None, storingDtype=None):
    if reactorType == 'cstV':
        if computationArchitechture == 'pt':
            jacobianComp = reactors_pt.CUDAFullJacobianConstantVTPY_PT(mech, torchComputationDtype=torch.float64, dataDevice=storingDevice, computationDevice=computationDevice, storageDtype=storingDtype)
        elif computationArchitechture == 'pb':
            jacobianComp = reactors_pb.CUDAFullJacobianConstantVTPY(mech, torchComputationDtype=torch.float64, dataDevice=storingDevice, computationDevice=computationDevice, storageDtype=storingDtype)
        else:
            raise NotImplementedError('computation architechture must be either pt or pb')
    elif reactorType == 'cstP':
        if computationArchitechture == 'pt':
            jacobianComp = reactors_pt.CUDAFullJacobianConstantPTPY_PT(mech, torchComputationDtype=torch.float64, dataDevice=storingDevice, computationDevice=computationDevice, storageDtype=storingDtype)
        elif computationArchitechture == 'pb':
            jacobianComp = reactors_pb.CUDAFullJacobianConstantPTPY(mech, torchComputationDtype=torch.float64, dataDevice=storingDevice, computationDevice=computationDevice, storageDtype=storingDtype)
        else:
            raise NotImplementedError('computation architechture must be either pt or pb')
    else:
        raise NotImplementedError('reactorType must be cstV or cstP')

    J_chem = batched_apply(jacobianComp.evaluate, X.data[:, 0].flatten(), X.data[:, 1].flatten(), X.data[:, 2:], batch_size=batch_size, device=computationDevice, dtype=torch.float64)
    return(serie(J_chem, ['J'], storingDevice=storingDevice, storingDtype=storingDtype))


def getTransformationJ_1(reactorType, mech, X, dt, computationArchitechture='pt', batch_size=15_000, computationDevice=None, storingDevice=None, storingDtype=None):
    J_chem = getChemicalJacobian(reactorType, mech, X, computationArchitechture, batch_size, computationDevice, storingDevice, storingDtype)
    J_transfo = - dt * J_chem.data
    n = X.nFeatures
    J_transfo[:, range(n), range(n)] += 1
    return(serie(J_transfo, ['J'], storingDevice=storingDevice, storingDtype=storingDtype))

def getScaledTransformationJ_1(reactorType, mech, X, Y, inputScaler, outputScaler, dt, computationArchitechture='pt', batch_size=15_000, computationDevice=None, storingDevice=None, storingDtype=None):
    J_phys = getTransformationJ_1(reactorType, mech, Y, dt, computationArchitechture, batch_size, computationDevice, storingDevice, storingDtype)

    J_1_out = outputScaler.J_1(outputScaler.transform(Y))
    J_in = inputScaler.J(X)

    J_scaled = J_in.unsqueeze(2) * J_phys.data * J_1_out.unsqueeze(1)
    return serie(J_scaled, ['J'], storingDevice=storingDevice, storingDtype=storingDtype)
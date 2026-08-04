import torch
from POLENE_GPU.dataHandler.serie import serie
from POLENE_GPU.misc.batchFunction import batched_apply

import kineticsGPU.cuda.kernels.per_thread.reactor_odes as reactors_pt
import kineticsGPU.cuda.kernels.per_block.reactor_odes_pb as reactors_pb


def _make_jacobian_computer(reactorType, computationArchitechture, mech, storingDevice, computationDevice, storingDtype):
    registry = {
        ('cstV', 'pt'): reactors_pt.CUDAFullJacobianConstantVTPY_PT,
        ('cstV', 'pb'): reactors_pb.CUDAFullJacobianConstantVTPY,
        ('cstP', 'pt'): reactors_pt.CUDAFullJacobianConstantPTPY_PT,
        ('cstP', 'pb'): reactors_pb.CUDAFullJacobianConstantPTPY,
    }
    key = (reactorType, computationArchitechture)
    if key not in registry:
        raise NotImplementedError(f"reactorType doit être 'cstV' ou 'cstP', computationArchitechture doit être 'pt' ou 'pb'. Reçu : {key}")
    return registry[key](mech, torchComputationDtype=torch.float64, dataDevice=storingDevice, computationDevice=computationDevice, storageDtype=storingDtype)


def getChemicalJacobian(reactorType, mech, X, computationArchitechture='pt', batch_size=15_000, computationDevice=None, storingDevice=None, storingDtype=None):
    jacobianComp = _make_jacobian_computer(reactorType, computationArchitechture, mech, storingDevice, computationDevice, storingDtype)

    def compute(T, P, Y):
        return jacobianComp.evaluate(T, P, Y)

    J_chem = batched_apply(compute, X.data[:, 0].flatten(), X.data[:, 1].flatten(), X.data[:, 2:], batch_size=batch_size, device=computationDevice, dtype=torch.float64)
    return serie(J_chem, ['J'], storingDevice=storingDevice, storingDtype=storingDtype)


def getTransformationJ_1(reactorType, mech, X, dt, computationArchitechture='pt', batch_size=15_000, computationDevice=None, storingDevice=None, storingDtype=None):
    jacobianComp = _make_jacobian_computer(reactorType, computationArchitechture, mech, storingDevice, computationDevice, storingDtype)

    def compute(T, P, Y):
        J = -dt * jacobianComp.evaluate(T, P, Y)
        n = J.shape[-1]
        J[:, range(n), range(n)] += 1
        return J

    J_transfo = batched_apply(compute, X.data[:, 0].flatten(), X.data[:, 1].flatten(), X.data[:, 2:], batch_size=batch_size, device=computationDevice, dtype=torch.float64)
    return serie(J_transfo, ['J'], storingDevice=storingDevice, storingDtype=storingDtype)


def getScaledTransformationJ_1(reactorType, mech, X, Y, inputScaler, outputScaler, dt, computationArchitechture='pt', batch_size=15_000, computationDevice=None, storingDevice=None, storingDtype=None):
    jacobianComp = _make_jacobian_computer(reactorType, computationArchitechture, mech, storingDevice, computationDevice, storingDtype)

    def compute(T, P, Yspec, x, y):
        J_phys = -dt * jacobianComp.evaluate(T, P, Yspec)
        n = J_phys.shape[-1]
        J_phys[:, range(n), range(n)] += 1
        J_1_out = outputScaler.J_1(outputScaler.transform(serie(y, Y.colName, storingDevice=storingDevice, storingDtype=storingDtype)))
        J_in = inputScaler.J(serie(x, X.colName, storingDevice=storingDevice, storingDtype=storingDtype))
        return J_in.unsqueeze(2) * J_phys * J_1_out.unsqueeze(1)

    J_scaled = batched_apply(compute, Y.data[:, 0].flatten(), Y.data[:, 1].flatten(), Y.data[:, 2:], X.data, Y.data, batch_size=batch_size, device=computationDevice, dtype=torch.float64)
    return serie(J_scaled, ['J'], storingDevice=storingDevice, storingDtype=storingDtype)



def getTransformationJ(reactorType, mech, Y, dt, computationArchitechture='pt', batch_size=15_000, computationDevice=None, storingDevice=None, storingDtype=None):
    jacobianComp = _make_jacobian_computer(reactorType, computationArchitechture, mech, storingDevice, computationDevice, storingDtype)

    def compute(T, P, Yspec):
        J = -dt * jacobianComp.evaluate(T, P, Yspec)
        n = J.shape[-1]
        J[:, range(n), range(n)] += 1
        return torch.linalg.inv(J)

    J_transfo = batched_apply(compute, Y.data[:, 0].flatten(), Y.data[:, 1].flatten(), Y.data[:, 2:], batch_size=batch_size, device=computationDevice, dtype=torch.float64)
    return serie(J_transfo, ['J'], storingDevice=storingDevice, storingDtype=storingDtype)


def getScaledTransformationJ(reactorType, mech, X, Y, inputScaler, outputScaler, dt, computationArchitechture='pt', batch_size=15_000, computationDevice=None, storingDevice=None, storingDtype=None):
    jacobianComp = _make_jacobian_computer(reactorType, computationArchitechture, mech, storingDevice, computationDevice, storingDtype)

    def compute(T, P, Yspec, x, y):
        J_phys = -dt * jacobianComp.evaluate(T, P, Yspec)
        n = J_phys.shape[-1]
        J_phys[:, range(n), range(n)] += 1
        J_phys = torch.linalg.inv(J_phys)
        J_out = outputScaler.J(serie(y, Y.colName, storingDevice=storingDevice, storingDtype=storingDtype))
        J_1_in = inputScaler.J_1(inputScaler.transform(serie(x, X.colName, storingDevice=storingDevice, storingDtype=storingDtype)))
        return J_out.unsqueeze(2) * J_phys * J_1_in.unsqueeze(1)

    J_scaled = batched_apply(compute, Y.data[:, 0].flatten(), Y.data[:, 1].flatten(), Y.data[:, 2:], X.data, Y.data, batch_size=batch_size, device=computationDevice, dtype=torch.float64)
    return serie(J_scaled, ['J'], storingDevice=storingDevice, storingDtype=storingDtype)

def getScaledTransformationJExplicit(reactorType, mech, X, Y, inputScaler, outputScaler, dt, computationArchitechture='pt', batch_size=15_000, computationDevice=None, storingDevice=None, storingDtype=None):
    jacobianComp = _make_jacobian_computer(reactorType, computationArchitechture, mech, storingDevice, computationDevice, storingDtype)

    def compute(T, P, Yspec, x, y):
        J_phys = dt * jacobianComp.evaluate(T, P, Yspec)
        n = J_phys.shape[-1]
        J_phys[:, range(n), range(n)] += 1
        J_out = outputScaler.J(serie(y, Y.colName, storingDevice=storingDevice, storingDtype=storingDtype))
        J_1_in = inputScaler.J_1(inputScaler.transform(serie(x, X.colName, storingDevice=storingDevice, storingDtype=storingDtype)))
        return J_out.unsqueeze(2) * J_phys * J_1_in.unsqueeze(1) #TODO: vérifier cela car il semblerai que la formule soit +- inversée, ce qui pourait expliquer les résultats un peu étranges rencontrés.

    J_scaled = batched_apply(compute, X.data[:, 0].flatten(), X.data[:, 1].flatten(), X.data[:, 2:], X.data, Y.data, batch_size=batch_size, device=computationDevice, dtype=torch.float64)
    return serie(J_scaled, ['J'], storingDevice=storingDevice, storingDtype=storingDtype)
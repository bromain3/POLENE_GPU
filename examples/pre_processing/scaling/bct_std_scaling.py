from POLENE_GPU.preProcessing.scaler.BCT import boxCoxScaler
from POLENE_GPU.preProcessing.scaler.stdScaler import stdScaler
import POLENE_GPU.databaseGeneration.models.cantera0D as cantera0D
import POLENE_GPU.databaseGeneration.databaseGenerator as databaseGenerator
from POLENE_GPU.misc.timer import Timer


import torch
import matplotlib.pyplot as plt

device = torch.device('cuda:0')
dtype = torch.float64


with Timer("Model creation"):
    model = cantera0D.cstP(
        kinetics='gri30.yaml'
    )
    dtbgen = databaseGenerator.DatabaseGenerator(model, storingDevice=device, storingDtype=dtype)


with Timer("Scaler creation"):
    scaler = boxCoxScaler() + stdScaler()

with Timer("Single trajectory"):
    X, Y, tX, tY = dtbgen.generateSingleTrajectory({'T': 1700, 'P': 101325, 'phi': 1.0})

with Timer("Scaler fitting"):
    scaler.fit(X)

with Timer("Scaler fit-transforming"):
    fittransform_results = scaler.fit_transform(X)

with Timer("Scaler transforming"):
    transform_results = scaler.transform(X)

with Timer("Scaler inverse transforming"):
    invtransform_results = scaler.inverse_transform(transform_results)

with Timer("Transform jacobian"):
    J = scaler.J(X)

with Timer("Inverse transform jacobian"):
    J_1 = scaler.J_1(transform_results)


print(f'Sample: {(J * J_1)[0]}\n{(J * J_1)[5]}\n{(J * J_1)[-1]}')

plt.plot(tX['t'].cpu().detach().numpy(), X['T'].cpu().detach().numpy(), label='original')
plt.plot(tX['t'].cpu().detach().numpy(), invtransform_results['T'].cpu().detach().numpy(), linestyle='--', label='transformed-unstransformed')
plt.legend()
plt.title('T')
plt.show()

plt.plot(tX['t'].cpu().detach().numpy(), transform_results['T'].cpu().detach().numpy(), label='transform')
plt.plot(tX['t'].cpu().detach().numpy(), fittransform_results['T'].cpu().detach().numpy(), linestyle='--', label='fit-transform')
plt.legend()
plt.title('T')
plt.show()

plt.plot(tX['t'].cpu().detach().numpy(), transform_results['P'].cpu().detach().numpy(), label='transform')
plt.plot(tX['t'].cpu().detach().numpy(), fittransform_results['P'].cpu().detach().numpy(), linestyle='--', label='fit-transform')
plt.legend()
plt.title('P')
plt.show()

plt.plot(tX['t'].cpu().detach().numpy(), transform_results['AR'].cpu().detach().numpy(), label='transform')
plt.plot(tX['t'].cpu().detach().numpy(), fittransform_results['AR'].cpu().detach().numpy(), linestyle='--', label='fit-transform')
plt.legend()
plt.title('AR')
plt.show()

plt.plot(tX['t'].cpu().detach().numpy(), transform_results['OH'].cpu().detach().numpy(), label='transform')
plt.plot(tX['t'].cpu().detach().numpy(), fittransform_results['OH'].cpu().detach().numpy(), linestyle='--', label='fit-transform')
plt.legend()
plt.title('OH')
plt.show()

plt.plot(tX['t'].cpu().detach().numpy(), X['AR'].cpu().detach().numpy(), label='original')
plt.plot(tX['t'].cpu().detach().numpy(), invtransform_results['AR'].cpu().detach().numpy(), linestyle='--', label='transformed-unstransformed')
plt.legend()
plt.title('AR')
plt.show()

plt.plot(tX['t'].cpu().detach().numpy(), X['P'].cpu().detach().numpy(), label='original')
plt.plot(tX['t'].cpu().detach().numpy(), invtransform_results['P'].cpu().detach().numpy(), linestyle='--', label='transformed-unstransformed')
plt.legend()
plt.title('P')
plt.show()
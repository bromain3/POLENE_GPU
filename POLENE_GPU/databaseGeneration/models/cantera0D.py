import cantera as ct
import numpy as np
from scipy.stats import qmc

def phi2Y(phi):
    MMCH4 = 16.04
    MMO2 = 31.999
    O2_N2 = 3.32

    mch4_mo2_st = 1 * MMCH4 / (2 * MMO2)
    Y = {'CH4': phi * mch4_mo2_st * 2 * MMO2, 'O2': 2 * MMO2, 'N2': 2 * MMO2 * O2_N2}
    return (Y)

class __canteraReactorGenerator:
    def __init__(self, kinetics='gri30.yaml', dtcfd=1e-6, tmax=1e-3, equilibriumTolerance=1e-4, cstSamplingdt=False, samplingRange=None, limitMaxTimeStep=True):
        self.cstSamplingdt = cstSamplingdt
        self.dtcfd = dtcfd
        self.tmax = tmax
        self.equilibriumTolerance = equilibriumTolerance
        self.kinetics = kinetics
        if samplingRange is None:
            self.samplingRange = {'T': [1600, 1800], 'P': 101325, 'phi': [0.7, 1.5]}
        else:
            self.samplingRange = samplingRange

        self.singleTrajFeaturesNames = 2 * [['T', 'P'] + list(ct.Solution(kinetics).species_names)] + 2 * [['t']]
        self.dtbTrajFeaturesNames = 2 * [['T', 'P'] + list(ct.Solution(kinetics).species_names)]
        self.samplingVariables = list(self.samplingRange.keys())
        self.limitMaxTimeStep = limitMaxTimeStep

    def _computeEquilibriumT(self, T, P, Y, reactorType):
        gas = ct.Solution(self.kinetics)
        gas.TPY = T, P, Y
        gas.equilibrate(reactorType)
        return gas.T

    def _integrate(self, reactorType, gas, Teq):
        targetsReactor = reactorType(gas)
        targetsNet = ct.ReactorNet([targetsReactor])

        featuresReactor = reactorType(gas)
        featuresNet = ct.ReactorNet([featuresReactor])
        if not self.cstSamplingdt and self.limitMaxTimeStep:
            featuresNet.max_time_step = self.dtcfd #TODO: activating and deactivating it to show the difference in post combustion part of the trajcetory

        X = []
        Y = []
        time_X = []
        time_Y = []

        # simulation
        while (featuresNet.time + self.dtcfd < self.tmax) and abs((Teq - featuresReactor.T) / Teq) > self.equilibriumTolerance:

            X.append([featuresReactor.T, featuresReactor.thermo.P] + list(featuresReactor.thermo.Y))
            time_X.append(featuresNet.time)

            targetsNet.advance(featuresNet.time + self.dtcfd)

            Y.append([targetsReactor.T, targetsReactor.thermo.P] + list(targetsReactor.thermo.Y))
            time_Y.append(targetsNet.time)

            if self.cstSamplingdt:
                featuresNet.advance(featuresNet.time + self.dtcfd)
            else:
                featuresNet.step()

        # store in numpy arrays to make the serialization / deserialization occuring during parallel processing faster
        X = np.array(X) #serie(X, ['T', 'P'] + list(gas.species_names), storingDevice=self.storingDevice, storingDtype=self.storingDtype)
        Y = np.array(Y) #serie(Y, ['T', 'P'] + list(gas.species_names), storingDevice=self.storingDevice, storingDtype=self.storingDtype)
        time_X = np.array(time_X) # serie(time_X, ['t'], storingDevice=self.storingDevice, storingDtype=self.storingDtype)
        time_Y = np.array(time_Y) #serie(time_Y, ['t'], storingDevice=self.storingDevice, storingDtype=self.storingDtype)

        return X, Y, time_X, time_Y

    def generateInitialConditionSamples(self, sampligMethode, nTrajectories):
        hypercubeSize = len(self.samplingRange)

        # checking if the variable is constant or not (for pressure for instance)
        const = []
        for var, r in self.samplingRange.items():
            const.append(not isinstance(r, (list, tuple, np.ndarray)))

        # Pseudo random engine generation and samples generation only for variables that are not constant
        sampler = getattr(qmc, sampligMethode)(d=const.count(False))
        sample = sampler.random(n=nTrajectories)

        # scaling samples
        l_bound = []
        u_bound = []
        for var, r in self.samplingRange.items():
            if isinstance(r, (list, tuple, np.ndarray)):
                l_bound.append(min(r))
                u_bound.append(max(r))

        sample = qmc.scale(sample, l_bound, u_bound)

        # reconstructing the full sample table
        res = np.zeros((nTrajectories, hypercubeSize))
        usedSamplingCols = 0
        for i in range(hypercubeSize):
            if const[i]:
                val = list(self.samplingRange.values())[i]
                res[:, i] = val
            else:
                res[:, i] = sample[:, usedSamplingCols]
                usedSamplingCols += 1

        res = np.array(res) # serie(res, list(self.samplingRange.keys()), storingDevice=self.storingDevice, storingDtype=self.storingDtype)
        return res

    def generateSingleTrajectory(self, initialConditions):
        raise NotImplementedError()

    def generateDatabaseTrajectoriy(self, initialConditions):
        X, Y, tX, ty = self.generateSingleTrajectory(initialConditions)
        return(X, Y)

class cstV(__canteraReactorGenerator):
    def generateSingleTrajectory(self, initialCondition):
        Tini, Pini, Yini = initialCondition['T'], initialCondition['P'], phi2Y(initialCondition['phi'])
        Teq = self._computeEquilibriumT(Tini, Pini, Yini, 'UV')
        gas = ct.Solution(self.kinetics)
        gas.TPY = Tini, Pini, Yini

        return self._integrate(ct.IdealGasReactor, gas, Teq)

class cstP(__canteraReactorGenerator):
    def generateSingleTrajectory(self, initialCondition):
        Tini, Pini, Yini = initialCondition['T'], initialCondition['P'], phi2Y(initialCondition['phi'])
        Teq = self._computeEquilibriumT(Tini, Pini, Yini, 'HP')
        gas = ct.Solution(self.kinetics)
        gas.TPY = Tini, Pini, Yini

        return self._integrate(ct.IdealGasConstPressureReactor, gas, Teq)
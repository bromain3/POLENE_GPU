from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np

from POLENE_GPU.dataHandler.serie import concat, serie



class DatabaseGenerator:

    def __init__(self, generationModel, storingDevice=None, storingDtype=None):
        self.checkGenerationModel(generationModel)
        self.generationModel = generationModel
        self.storingDevice = storingDevice
        self.storingDtype = storingDtype

    @staticmethod
    def checkGenerationModel(model):
        if not hasattr(model, 'generateInitialConditionSamples'):
            raise NotImplementedError('To be used within the database generator API, the passed generation model must implement a generateInitialConditionSamples methode')
        if not hasattr(model, 'generateSingleTrajectory'):
            raise NotImplementedError('To be used within the database generator API, the passed generation model must implement a generateSingleTrajectory methode')
        if not hasattr(model, 'generateDatabaseTrajectoriy'):
            raise NotImplementedError('To be used within the database generator API, the passed generation model must implement a generateDatabaseTrajectoriy methode')
        if not hasattr(model, 'singleTrajFeaturesNames'):
            raise NotImplementedError('To be used within the database generator API, the passed generation model must define a singleTrajFeaturesNames attribute')
        if not hasattr(model, 'dtbTrajFeaturesNames'):
            raise NotImplementedError('To be used within the database generator API, the passed generation model must define a dtbTrajFeaturesNames attribute')
        if not hasattr(model, 'samplingVariables'):
            raise NotImplementedError('To be used within the database generator API, the passed generation model must define a samplingVariables attribute')

    def generateInitialConditionSamples(self, sampligMethode, nTrajectories):
        res = serie(self.generationModel.generateInitialConditionSamples(sampligMethode, nTrajectories), self.generationModel.samplingVariables, storingDevice=self.storingDevice, storingDtype=self.storingDtype)
        return(res)

    def generateSingleTrajectory(self, initialCondition):
        res = self.generationModel.generateSingleTrajectory(initialCondition)
        res = [serie(dt, self.generationModel.singleTrajFeaturesNames[i], storingDevice=self.storingDevice, storingDtype=self.storingDtype) for i, dt in enumerate(res)]
        return(res)


    def generateDataset(self, initialConditionSamples, CPUparallelProcessing=False):
        if CPUparallelProcessing:
            with Parallel(n_jobs=CPUparallelProcessing, backend='loky') as parallel:
                database_trajectories = parallel(delayed(self.generationModel.generateDatabaseTrajectoriy)(s) for s in tqdm(initialConditionSamples, total=initialConditionSamples.nSamples))
            # TODO: find a way to ensure all workers are killed
            #parallel._backend.abort_everything()
            #del parallel
        else:
            database_trajectories = []
            for s in tqdm(initialConditionSamples):
                database_trajectories.append(self.generationModel.generateDatabaseTrajectoriy(s))

        # concat on np arrays is 1000 times faster than with series
        n_outputs = len(self.generationModel.dtbTrajFeaturesNames)
        result = []
        for i in range(n_outputs):
            arrays = [traj[i] for traj in database_trajectories]  # liste de np.array
            merged = np.concatenate(arrays, axis=0)  # UN SEUL concat
            result.append(serie(
                merged,
                self.generationModel.dtbTrajFeaturesNames[i],
                storingDevice=self.storingDevice,
                storingDtype=self.storingDtype
            ))

        return result

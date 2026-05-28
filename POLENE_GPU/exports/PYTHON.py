import os
import joblib
import json
import torch
#TODO: create saving path if it does not exist otherwise it returns runtime_error
class PYTHONexport:

    tree = {'metaData': 'meta', 'model': 'models', 'model_scaler': 'model'}

    #TODO: in the end, database metadata (keptColumns) could be associated to the model itself as it relates to how it has been trained
    def __init__(self, model, keptColumns):
        self.model = model
        self.keptColumns = keptColumns

    def __saveMetaData(self, rootPath):
        keptColumns = self.keptColumns
        js = {'columns': keptColumns}
        with open(os.path.join(rootPath, PYTHONexport.tree['metaData'], 'meta.json'), 'w') as f:
            json.dump(js, f)

    def __saveModels(self, rootPath, savingDevice='cpu'):
        # TODO: modify saving functions so that every python model capacity can be used in CEDRE
        self.model = self.model.to(savingDevice)
        self.model.eval()
        joblib.dump(self.model, os.path.join(rootPath, PYTHONexport.tree['model_scaler'], f'model.pkl'))


    def save(self, rootPath, savingDevice='cpu'):
        if not os.path.exists(rootPath):
            os.makedirs(rootPath)
        for k, v in PYTHONexport.tree.items():
            if not os.path.exists(os.path.join(rootPath, v)):
                os.makedirs(os.path.join(rootPath, v))

        self.__saveMetaData(rootPath)
        self.__saveModels(rootPath, savingDevice)

    def printSavingTree(self):
        #TODO: implement the function
        raise NotImplementedError('This function is not yet implemented')


def loadPYTHONexportFolder(rootPath):
    with open(os.path.join(rootPath, PYTHONexport.tree['metaData'], 'meta.json'), 'r') as f:
        metas = json.load(f)

    model = joblib.load(os.path.join(rootPath, PYTHONexport.tree['model_scaler'], f'model.pkl'))

    return(metas['columns'], model)
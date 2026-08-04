import os
import joblib
import json
import torch
from types import SimpleNamespace


class PYTHONexport:

    tree = {'metaData': 'meta', 'model': 'models', 'model_scaler': 'model'}

    def __init__(self, model, keptColumns, outputColumns=None, fictiveSpecies=None):
        self.model = model
        self.keptColumns = keptColumns
        # colonnes réellement prédites par le modèle (peut différer de keptColumns
        # si une espèce fictive/déduite a été retirée des sorties)
        self.outputColumns = outputColumns if outputColumns is not None else keptColumns
        # nom de l'espèce non prédite, à reconstruire comme 1 - somme(autres Y)
        self.fictiveSpecies = fictiveSpecies

    def __saveMetaData(self, rootPath):
        js = {
            'columns': self.keptColumns,
            'outputColumns': self.outputColumns,
            'fictiveSpecies': self.fictiveSpecies,
        }
        with open(os.path.join(rootPath, PYTHONexport.tree['metaData'], 'meta.json'), 'w') as f:
            json.dump(js, f)

    def __saveModels(self, rootPath, savingDevice='cpu'):
        self.model = self.model.to(savingDevice)
        self.model.eval()
        joblib.dump(self.model, os.path.join(rootPath, PYTHONexport.tree['model_scaler'], 'model.pkl'))

    def save(self, rootPath, savingDevice='cpu'):
        if not os.path.exists(rootPath):
            os.makedirs(rootPath)
        for k, v in PYTHONexport.tree.items():
            if not os.path.exists(os.path.join(rootPath, v)):
                os.makedirs(os.path.join(rootPath, v))

        self.__saveMetaData(rootPath)
        self.__saveModels(rootPath, savingDevice)

    def printSavingTree(self):
        raise NotImplementedError('This function is not yet implemented')


def loadPYTHONexportFolder(rootPath):
    with open(os.path.join(rootPath, PYTHONexport.tree['metaData'], 'meta.json'), 'r') as f:
        metas = json.load(f)

    model = joblib.load(os.path.join(rootPath, PYTHONexport.tree['model_scaler'], 'model.pkl'))

    # rétro-compatibilité avec les anciens exports sans ces clés
    outputColumns = metas.get('outputColumns', metas['columns'])
    fictiveSpecies = metas.get('fictiveSpecies', None)

    return SimpleNamespace(
        columns=metas['columns'],
        outputColumns=outputColumns,
        fictiveSpecies=fictiveSpecies,
        model=model,
    )
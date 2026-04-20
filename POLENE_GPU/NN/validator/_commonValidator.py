import torch

class _commonValidator:
    _IMPLEMENTED_METRICS = ['MSE', 'MAE', 'rMAE', 'rMSE']
    _METRICS_UNIT = {'MSE': '', 'MAE': '', 'rMAE':  '%', 'rMSE': '%'}
    METRIC_EPS = 1e-30

    def __init__(self, metric='MAE', validationDtype=None, validationDevice=None):
        self.validationDtype = validationDtype
        self.validationDevice = validationDevice
        self._checkMetric(metric)
        self.metric = metric

    def _checkInput(self, model, X, Y):
        if not hasattr(model, 'inputScaler'):
            raise AttributeError('For being validated, the model must store the attribute model.inputScaler')
        if not hasattr(model, 'outputScaler'):
            raise AttributeError('For being validated, the model must store the attribute model.outputScaler')
        if model.inputScaler is None:
            raise AttributeError('For being validated, the model must store the input scaler in model.inputScaler')
        if model.outputScaler is None:
            raise AttributeError('For being validated, the model must store the input scaler in model.outputScaler')

    def _checkMetric(self, metric):
        if metric not in _commonValidator._IMPLEMENTED_METRICS:
            raise NotImplementedError(f'the metric {metric} is not implemented. Available metrics are: {_commonValidator._IMPLEMENTED_METRICS}')

    def MSE(self, X, Y):
        with torch.no_grad():
            msePerCol = (X - Y).pow(2).mean(dim=0)
            return msePerCol, msePerCol.mean().item()

    def MAE(self, X, Y):
        with torch.no_grad():
            maePerCol = (X - Y).abs().mean(dim=0)
            return maePerCol, maePerCol.mean().item()

    def rMAE(self, X, Y):
        with torch.no_grad():
            rmaePerCol = 2 * ((X - Y).abs() / (X.abs() + Y.abs() + _commonValidator.METRIC_EPS)).mean(dim=0)
            return rmaePerCol, rmaePerCol.mean().item()

    def rMSE(self, X, Y):
        with torch.no_grad():
            rmsePerCol = 4 * ((X - Y).pow(2) / (X + Y + _commonValidator.METRIC_EPS).pow(2)).mean(dim=0)
            return rmsePerCol, rmsePerCol.mean().item()

    def computeStats(self, preds, Y):
        metricFunc = getattr(self, self.metric)
        perColStats, globalStats = metricFunc(preds, Y)
        return perColStats, globalStats

    def printStats(self, perColStats, globalStats):
        perColStats = {k:v for k,v in perColStats.items()}
        statsMessage = ('One step prediction statistics:\n'
                        f'\tGlobal {self.metric}: {globalStats} {_commonValidator._METRICS_UNIT[self.metric]}\n'
                        '\tStatistics per column:\n'
                        f'\t{perColStats}')
        print(statsMessage)

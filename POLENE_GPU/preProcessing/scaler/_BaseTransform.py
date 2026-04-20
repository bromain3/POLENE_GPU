from abc import ABC, abstractmethod

class _BaseTransform(ABC):

    NON_SPECIES_COLS = ['T', 'P']

    def __add__(self, other):
        """
        Compose deux scalers en pipeline.

        Exemple :
            scaler = stdScaler() + boxCoxScaler()
            scaler.fit_transform(x)
        """
        from POLENE_GPU.preProcessing.scaler._PipelineTransform import _PipelineTransform  # for preventing circular import
        if isinstance(self, _PipelineTransform):
            left_steps = self.steps
        else:
            left_steps = [self]

        if isinstance(other, _PipelineTransform):
            right_steps = other.steps
        else:
            right_steps = [other]

        return _PipelineTransform(left_steps + right_steps)

    @abstractmethod
    def fit(self, x):
        raise NotImplementedError

    @abstractmethod
    def transform(self, x):
        raise NotImplementedError

    @abstractmethod
    def transformTorch(self, x):
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, x):
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, x):
        raise NotImplementedError

    @abstractmethod
    def inverse_transformTorch(self, x):
        raise NotImplementedError

    @abstractmethod
    def J(self, x):
        raise NotImplementedError

    @abstractmethod
    def J_1(self, x):
        raise NotImplementedError
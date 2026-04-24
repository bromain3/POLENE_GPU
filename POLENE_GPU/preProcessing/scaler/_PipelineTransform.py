from POLENE_GPU.preProcessing.scaler._BaseTransform import _BaseTransform

import torch


class _PipelineTransform(_BaseTransform):
    def __init__(self, steps: list):
        self.steps = steps

    def __add__(self, other):
        if isinstance(other, _PipelineTransform):
            return _PipelineTransform(self.steps + other.steps)
        return _PipelineTransform(self.steps + [other])

    def __radd__(self, other):
        if isinstance(other, _PipelineTransform):
            return _PipelineTransform(other.steps + self.steps)
        return _PipelineTransform([other] + self.steps)

    def fit(self, x):
        current = x
        for step in self.steps:
            step.fit(current)
            current = step.transform(current)
        return self

    def transform(self, x):
        current = x
        for step in self.steps:
            current = step.transform(current)
        return current

    def transformTorch(self, x):
        current = x
        for step in self.steps:
            current = step.transformTorch(current)
        return current

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        current = x
        for step in reversed(self.steps):
            current = step.inverse_transform(current)
        return current

    def inverse_transformTorch(self, x):
        current = x
        for step in reversed(self.steps):
            current = step.inverse_transformTorch(current)
        return current

    def J(self, x):
        #TODO: raise an error if the jacobian is not diagonal
        intermediates = [x]
        current = x
        for step in self.steps[:-1]:
            current = step.transform(current)
            intermediates.append(current)

        J_total = self.steps[-1].J(intermediates[-1])

        for step, x_in in zip(reversed(self.steps[:-1]), reversed(intermediates[:-1])):
            J_total = J_total * step.J(x_in)

        return J_total

    def J_1(self, x):
        #TODO: raise an error if the jacobian is not diagonal
        intermediates = [x]
        current = x
        for step in reversed(self.steps[1:]):
            current = step.inverse_transform(current)
            intermediates.append(current)

        J_total = self.steps[-1].J_1(intermediates[0])
        for step, x_in in zip(reversed(self.steps[:-1]), intermediates[1:]):
            J_total = J_total * step.J_1(x_in)

        return J_total

    def __repr__(self) -> str:
        steps_repr = " → ".join(type(s).__name__ for s in self.steps)
        return f"_PipelineTransform([{steps_repr}])"
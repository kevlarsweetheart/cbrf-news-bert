import numpy as np

from bert.record import Record, parse_annotation
from bert.visitor import Visitor


class Weight(Record):
    __attributes__ = ['shape', 'dtype', 'array']

    def empty(self):
        return self.replace(array=None)

    @property
    def is_empty(self):
        return self.array is None

    @property
    def is_id(self):
        return type(self.array) is int


class Module(Record):
    @property
    def weights(self):
        visitor = WeightsVisitor()
        visitor(self)
        return visitor.weights


class Linear(Module):
    __attributes__ = ['weight', 'bias']
    __annotations__ = {
        'weight': Weight,
        'bias': Weight
    }

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.in_dim, self.out_dim = self.weight.shape

    def __call__(self, _input):
        shape = _input.shape
        _input = _input.reshape(-1, self.in_dim)
        output = np.matmul(_input, self.weight.array) + self.bias.array

        shape = shape[:-1] + (self.out_dim,)
        return output.reshape(*shape)


class ReLU(Module):
    def __call__(self, input):
        return input.clip(0)


class BatchNorm1d(Module):
    __attributes__ = ['weight', 'bias', 'mean', 'std']
    __annotations__ = {
        'weight': Weight,
        'bias': Weight,
        'mean': Weight,
        'std': Weight
    }

    def __call__(self, _input):
        # input is N x C x L, do ops on C
        _input = _input.swapaxes(2, 1)
        output = (
            (_input - self.mean.array)
            / self.std.array
            * self.weight.array
            + self.bias.array
        )
        return output.swapaxes(2, 1)  # recover shape


class Embedding(Module):
    __attributes__ = ['weight']
    __annotations__ = {
        'weight': Weight
    }

    def __init__(self, weight):
        self.weight = weight
        _, self.dim = self.weight.shape

    def __call__(self, _input):
        shape = _input.shape
        _input = _input.flatten()
        weight = self.weight.array[input]
        return weight.reshape(*shape, self.dim)


class ModuleVisitor(Visitor):
    def visit_Weight(self, item):
        return item

    def visit_Module(self, item):
        args = []
        for key in item.__attributes__:
            value = getattr(item, key)
            annotation = item.__annotations__.get(key)
            if annotation and value is not None:
                _, repeatable, _ = parse_annotation(annotation)
                if repeatable:
                    value = [self.visit(_) for _ in value]
                else:
                    value = self.visit(value)
            args.append(value)
        return type(item)(*args)


class WeightsVisitor(ModuleVisitor):
    def __init__(self):
        self.weights = []

    def visit_Weight(self, item):
        self.weights.append(item)
        return item

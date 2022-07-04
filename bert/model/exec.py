import torch

from bert.visitor import Visitor
from bert.exec import model as exec


class ExecVisitor(Visitor):
    def visit_Parameter(self, item):
        return self.visit(item.data)

    def visit_Tensor(self, item):
        array = item.detach().numpy()
        return exec.Weight(
            array.shape,
            array.dtype.name,
            array
        )

    def visit_Linear(self, item):
        # in torch linear is xA^T + b
        weight = item.weight.transpose(1, 0)
        return exec.Linear(
            self.visit(weight),
            self.visit(item.bias)
        )

    def visit_ReLU(self, item):
        return exec.ReLU()

    def visit_BatchNorm1d(self, item):
        running_std = torch.sqrt(item.running_var + item.eps)
        return exec.BatchNorm1d(
            self.visit(item.weight),
            self.visit(item.bias),
            self.visit(item.running_mean),
            self.visit(running_std),
        )

    def visit_Embedding(self, item):
        return exec.Embedding(
            self.visit(item.weight)
        )


class ExecMixin:
    # super stange error if as_exec property
    # torch Module does some magic
    def to_exec(self):
        visitor = ExecVisitor()
        return visitor(self)

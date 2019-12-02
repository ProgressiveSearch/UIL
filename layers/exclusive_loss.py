# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from torch import nn, autograd


class Exclusive(autograd.Function):
    def __init__(self, V):
        super(Exclusive, self).__init__()
        self.V = V
        # self.C = C

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.V.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = grad_outputs.mm(self.V) if self.needs_input_grad[0] else None
        for x, y in zip(inputs, targets):
            self.V[y] = F.normalize((self.V[y] + x) / 2, p=2, dim=0)
            # self.C[y] = (self.C[y] + x) / 2
        return grad_inputs, None


class ExLoss(nn.Module):
    def __init__(self, num_features, num_classes, t=1.0,
                 weight=None):
        super(ExLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.t = t
        self.weight = weight
        self.register_buffer('V', torch.zeros(num_classes, num_features))
        # self.register_buffer('C', torch.zeros(num_classes, num_features))
        self.register_parameter('C', nn.Parameter(torch.zeros(num_classes, num_features)))

    def forward(self, inputs, targets):
        outputs = Exclusive(self.V)(inputs, targets) * self.t

        # log_probs = F.log_softmax(outputs, dim=1)
        # targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        # targets = targets.cuda()
        # targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # loss = (-targets * log_probs).mean(0).sum()
        # return loss, outputs
        loss = F.cross_entropy(outputs, targets, weight=self.weight)
        return loss, outputs

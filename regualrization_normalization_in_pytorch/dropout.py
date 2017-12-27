import torch
import torch.nn as nn
import torch.nn.Parameter as Parameter
from torch.autograd import Variable



class DropConnect(nn.Module):
    def __init__(self, input_size, output_size, p=0.5):
        super(DropConnect, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.p = p
        self.weight = Parameter(torch.FloatTensor(input_size, output_size).normal_(mean=0, std=0.01), requires_grad=True)
        self.bias = Parameter(torch.FloatTensor(1, output_size).zero_(), requires_grad=True)

    def forward(self, x):
        self.mask = Variable(torch.FloatTensor(self.input_size, self.output_size).bernoulli_(self.p), requires_grad=False)
        batch_size = x.mm(self.weight * self.mask).size()[0]
        output = x.mm(self.weight * self.mask) + self.bias.repeat(batch_size, 1)
        return output


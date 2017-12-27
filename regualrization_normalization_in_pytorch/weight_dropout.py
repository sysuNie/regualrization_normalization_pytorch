import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def _setup(self):
        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = Variable(torch.ones(raw_w.size(0), 1))
                # if raw_w.is_cuda: mask = mask.cuda()
                mask = F.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = F.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)



if __name__ == '__main__':
    x = Variable(torch.randn(3, 4, 5))

    print('Testing WeightDrop with Linear')

    dfc =WeightDrop(nn.Linear(5, 10), ['weight'], dropout=0.2)

    print('Testing WeightDrop with LSTM')

    dRNN = WeightDrop(torch.nn.LSTM(5, 10), ['weight_hh_l0'], dropout=0.2)


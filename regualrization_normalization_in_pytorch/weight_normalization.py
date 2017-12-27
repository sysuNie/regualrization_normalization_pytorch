import torch.utils.hooks as hooks
from torch.nn.parameter import Parameter

'''
   Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
'''
class WeightNorm(object):
    def __init__(self, name, dim):
        self.name =name
        self.dim = dim

    def computer_weight(self, module):
        g = getattr(module, self.name+'_g')
        v = getattr(module, self.name+'_v')
        return (g/self.norm(v))*v

    def norm(self, p):
        """Computes the norm over all dimensions except dim"""
        if self.dim is None:
            return p.norm()
        if self.dim != 0:
            p = p.transpose(0, self.dim)
        output_size = (p.size(0),)+(1,)*(p.dim()-1)  # delete dim dimension
        p = p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
        if self.dim != 0:
            p = p.transpose(0, self.dim)
        return p

    @staticmethod
    def apply(modula, name, dim):
        fn = WeightNorm(name, dim)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + '_g', Parameter(fn.norm(weight).data))
        module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, fn.computer_weight(module))

        handle = hooks.RemovableHandle(module._forward_pre_hooks)
        module._forward_pre_hooks[handle.id] = fn
        fn.handle = handle

        return  fn

    def __call__(self, module, inputs):
        setattr(module, self.name, self.computer_weight(module))

'''
   example
'''

def weight_norm(module, name='weight', dim=0):

    WeightNorm.apply(module, name, dim)
    return module

'''  
    >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        Linear (20 -> 40)
    >>> m.weight_g.size()
        torch.Size([40, 1])
    >>> m.weight_v.size()
        torch.Size([40, 20])
'''







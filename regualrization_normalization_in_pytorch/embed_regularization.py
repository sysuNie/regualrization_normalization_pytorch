import numpy as np

import torch
from torch.autograd import Variable

'''
V = 3
h = 4
bptt = 10
batch_size = 2

embed = torch.nn.Embedding(V, h)

#print embed.weight
###
Parameter containing:
-0.3689  2.3818 -2.0666  1.9062
 0.3776  0.7572  0.9740 -0.0558
-2.4540  1.1843 -1.2406  0.1258
[torch.FloatTensor of size 3x4]
###

words = np.random.random_integers(low=0, high=V-1, size=(batch_size, bptt))
words = Variable(torch.LongTensor(words))

#print words
###
   Variable containing:
    2     0     2     1     1     1     2     1     0     2
    1     1     2     1     2     0     1     0     0     0
    [torch.LongTensor of size 2x10]
###

# If dropout
dropout = 0.1
mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1-dropout).expand_as(embed.weight)/(1-dropout)
# print mask  mask is a random matrix like bleow
###
    1.1111  1.1111  1.1111  1.1111    1.1111  1.1111  1.1111  1.1111
    0.0000  0.0000  0.0000  0.0000    1.1111  1.1111  1.1111  1.1111
    1.1111  1.1111  1.1111  1.1111    1.1111  1.1111  1.1111  1.1111
    [torch.FloatTensor of size 3x4]   [torch.FloatTensor of size 3x4]
###

masked_embed_weight = Variable(mask) * embed.weight
padding_idx = embed.padding_idx
if padding_idx is None:
    padding_idx = -1

orignX = embed(words)
X = embed._backend.Embedding.apply(words, masked_embed_weight,
                                   padding_idx, embed.max_norm, embed.norm_type,
                                   embed.scale_grad_by_freq, embed.sparse)
'''

# function

def embedded_dropout(embed, words, dropout=0.1,):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight

    else:
        masked_embed_weight = embed.weight


    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = embed._backend.Embedding.apply(words, masked_embed_weight,
                                       padding_idx, embed.max_norm, embed.norm_type,
                                       embed.scale_grad_by_freq, embed.sparse)

    return X


if __name__ == '__main__':
    V = 10
    h = 4
    bptt = 10
    batch_size = 2

    embed = torch.nn.Embedding(V, h)

    words = np.random.random_integers(low=0, high=V - 1, size=(batch_size, bptt))
    words = Variable(torch.LongTensor(words))

    origX = embed(words)
    mask_X = embedded_dropout(embed, words)
    # scale_X = embedded_dropout(embed, words, dropout=0, scale=True)
    # mask_scale_X = embedded_dropout(embed, words, scale=True)
    print origX
    print mask_X





# import torch
import torch
import torch.nn as nn
from torch.tensor import Tensor as T
# import numpy
import numpy as np

class Prototyp(nn.Module):
    
    def __init__(self, nh, K):
        super(Prototyp, self).__init__()
        # create tensors
        self.V = nn.Parameter(T(np.random.uniform(-0.2, 0.2, size=(nh, nh))))

    def forward(self, h, e, u):
        e = torch.softmax(e, dim=-1).view(*e.size(), 1)
        return torch.tanh(self.V @ u) + (h * e).sum(-2).permute(1, 0)
        

class CoupledAttention(nn.Module):
    
    def __init__(self, nh, K, dropout_rate=0.3):
        super(CoupledAttention, self).__init__()
        # attention gated-recurrent-unit
        self.attention = nn.GRU(2*K, 2*K, bias=False, batch_first=True)
        self.r0 = nn.Parameter(T(np.random.uniform(-0.2, 0.2, size=2*K)))
        # feature weight vector
        self.v = nn.Parameter(T(np.random.uniform(-0.2, 0.2, size=2*K)))
        # weight tensors
        self.G_a = nn.Parameter(T(np.random.uniform(-0.2, 0.2, size=(K, 1, nh, nh))))
        self.G_o = nn.Parameter(T(np.random.uniform(-0.2, 0.2, size=(K, 1, nh, nh))))
        # dropout
        self.G_a_drop = nn.Dropout(dropout_rate)
        self.G_o_drop = nn.Dropout(dropout_rate)

    def forward(self, h, u_a, u_o):
        # f = h @ G @ u
        f = lambda G, u: ((h @ G) * u.permute(1, 0).unsqueeze(1)).sum(-1)
        # apply f
        f1, f2 = f(self.G_a_drop(self.G_a), u_a), f(self.G_o_drop(self.G_o), u_o)
        beta = torch.cat((f1, f2), dim=0)
        beta = torch.tanh(beta)
        beta = beta.flatten(2).permute(1, 2, 0)
        # pass through gru
        r, _ = self.attention(beta, self.r0.repeat(1, beta.size(0), 1))
        # return weighted sum of features
        return r, r @ self.v


class CMLA(nn.Module):

    def __init__(self, nh, nc, vs, de, cs, K, l, dropout_rate=0.3, pad_id=0):
        """
        Parameters:
            nh:     hidden layer dimension
            nc:     number of classes
            vs:     vocab size
            de:     word embedding dimension
            cs:     word context window size
            K:      sequence embedding size
            l:      number of attention layers
            pad_id: padding index in 
        """
        super(CMLA, self).__init__()
        # save configuration
        self.config = {'nh':nh, 'nc': nc, 'vs': vs, 'de': de, 'cs': cs, 'K': K, 'l': l, 'dropout_rate': dropout_rate, 'pad_id': pad_id}
        # create embedding layer
        self.embedding = nn.Embedding(vs, de, padding_idx=pad_id)
        self.embeddGRU = nn.GRU(cs * de, nh, batch_first=True)
        self.embedd_h0 = nn.Parameter(torch.zeros(nh))
        # aspect and opinion attention layer
        self.aspect_attention = CoupledAttention(nh, K, dropout_rate)
        self.opinion_attention = CoupledAttention(nh, K, dropout_rate)
        # aspect and opinion prototyps
        self.aspect_proto = Prototyp(nh, K)
        self.opinion_proto = Prototyp(nh, K)
        # initial prototyp vectors
        self.aspect_u0 = nn.Parameter(T(np.random.uniform(-0.2, 0.2, size=(nh, 1))))
        self.opinion_u0 = nn.Parameter(T(np.random.uniform(-0.2, 0.2, size=(nh, 1))))
        # classification layers
        self.aspect_classifier = nn.Linear(2*K, nc)
        self.opinion_classifier = nn.Linear(2*K, nc)
        # dropout
        self.aspect_dropout = nn.Dropout(dropout_rate)
        self.opinion_dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):

        pad_id, cs = self.config['pad_id'], self.config['cs']
        # make context window
        pad = torch.LongTensor([pad_id] * input_ids.size(0) * (cs//2)).view(-1, 1)
        x = torch.cat((pad, input_ids, pad), dim=-1)
        x = torch.stack((x.roll(1, dims=-1), x, x.roll(-1, dims=-1)), dim=-1)
        x = x[..., 1:-1, :]
        # embedd input and flatten context-window
        x = self.embedding(x).flatten(-2)
        h, _ = self.embeddGRU(x, self.embedd_h0.repeat(1, x.size(0), 1))

        aspect_r, opinion_r = 0, 0
        aspect_ui, opinion_ui = self.aspect_u0, self.opinion_u0
        # pass through layers
        for _ in range(self.config['l']):
            # apply coupled attention
            aspect_ri, aspect_ei = self.aspect_attention(h, aspect_ui, opinion_ui)
            opinion_ri, opinion_ei = self.opinion_attention(h, aspect_ui, opinion_ui)
            # update attention vectors
            aspect_r += aspect_ri
            opinion_r += opinion_ri
            # update prototyp-vectors
            aspect_ui = self.aspect_proto(h, aspect_ei, aspect_ui)
            opinion_ui = self.aspect_proto(h, opinion_ei, opinion_ui)

        # pass through classifier
        aspect_logits = self.aspect_classifier(self.aspect_dropout(aspect_r))
        opinion_logits = self.opinion_classifier(self.opinion_dropout(opinion_r))

        # return logits
        return aspect_logits, opinion_logits


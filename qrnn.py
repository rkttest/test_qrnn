#coding:utf-8

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class FOPooling(nn.Module):
    def __init__(self, dropout_p=0.2):
        super(FOPooling, self).__init__()
        self.sigmoid = nn.functional.sigmoid
        self.tanh = nn.functional.tanh
        self.dropout_p = dropout_p
        
    def forward(self, Y, init_c, init_h):
        n_seq = Y.size()[0]
        F, Z, O = Y.chunk(3, dim=2)
        F = self.sigmoid(F)
        Z = self.tanh(Z)
        O = self.sigmoid(O)

        F = self.zoneout(F)
        
        ZF = torch.mul(1-F, Z)
        h_out = [init_h]
        c_out = [init_c]
        for t in range(n_seq):
            c = torch.add(torch.mul(F[t], c_out[-1]), ZF[t])
            h = torch.mul(c, O[t])
            h_out.append(h)
            c_out.append(c)

        h_out = torch.cat(h_out)
        return h_out

    def zoneout(self, F):
        F = 1 - nn.functional.dropout(1 - F, p=self.dropout_p)
        return F
            

        
class QRNN(nn.Module):

    def __init__(self, word_len=20, dict_size=60, hidden_size=64,
                 n_layers=2, dropout_p=0.2, kernel_size=1):
        super(QRNN, self).__init__()
        self.n_layers = n_layers
        self.conv1 = nn.Conv1d(hidden_size, hidden_size*3, kernel_size) #B * C * L -> B * C_out * L_out
        self.embed = nn.Embedding(dict_size, hidden_size, padding_idx=None)
        self.fopooling = FOPooling(dropout_p=dropout_p)
        
    def forward(self, X, init_h, init_c):
        x = self.embed(X) #X = B * N -> x = B * N * H
        x = x.transpose(1, 2)
        for _ in range(self.n_layers):
            x = self.conv1(x)
            x = x.transpose(1, 2)
            x = self.fopooling(x, init_h, init_c)

        return x


def main():
    #入力データは 20data * 50wordlen * 60dictsize * 64hiddensize
    X = np.stack([np.arange(50) for _ in range(20)])
    X += np.random.randint(0, 3, X.shape)
    X = Variable(torch.from_numpy(X))

    init_h = Variable(torch.zeros(20, 64))
    init_c = Variable(torch.zeros(20, 64))
    
    qrnn = QRNN()
    hidden_out = qrnn(X, init_h, init_c)

    
if __name__=="__main__":
    main()

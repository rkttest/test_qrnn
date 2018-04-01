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
        # Y = N * B * 3H
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

        h_out = torch.stack(h_out[1:])
        return h_out, c_out

    def zoneout(self, F):
        F = 1 - nn.functional.dropout(1 - F, p=self.dropout_p)
        return F
            

        
class QRNN(nn.Module):

    def __init__(self, hidden_size=64,
                 n_layers=2, dropout_p=0.2, kernel_size=1):
        super(QRNN, self).__init__()
        self.n_layers = n_layers
        self.conv1 = nn.Conv1d(hidden_size, hidden_size*3, kernel_size) #B * C * L -> B * C_out * L_out
        self.fopooling = FOPooling(dropout_p=dropout_p)
        
    def forward(self, x, init_h, init_c, target=None):
        # x = N * B * H
            
        for _ in range(self.n_layers):
            h_out, c_out = self.conv2pool(x, init_h, init_c)
            init_c = c_out[-1]
            init_h = h_out[-1]
            x = h_out
        return x, c_out


    def conv2pool(self, x, init_h, init_c):
        x = x.transpose(1, 2) #x = N * H * B
        x = self.conv1(x)
        x = x.transpose(1, 2) #x = N * B * H
        h_out, c_out = self.fopooling(x, init_h, init_c)
        return h_out, c_out


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.functional.softmax
        
    def forward(self, input_encode, target_encode):
        #input_encode = N * B * H
        #target_encode = M * B * H
        #matrix = B * M * N
        #attn = B * M * N
        input_encode = input_encode.transpose(0, 1).transpose(1, 2)
        target_encode = target_encode.transpose(0, 1)
        matrix = torch.bmm(target_encode, input_encode)
        attn = self.softmax(matrix, dim=2)
        return attn

def main():
    #入力データは 20data * 50wordlen * 60dictsize * 64hiddensize
    X = np.stack([np.arange(50) for _ in range(20)])
    Y = np.stack([np.arange(50) for _ in range(20)]) + 2
    X += np.random.randint(0, 3, X.shape)
    X = Variable(torch.from_numpy(X))
    Y = Variable(torch.from_numpy(Y))
    init_h = Variable(torch.zeros(20, 64))
    init_c = Variable(torch.zeros(20, 64))

    embed = nn.Embedding(60, 64,
                         padding_idx=None)
    X = embed(X).transpose(0, 1)
    Y = embed(Y).transpose(0, 1)
    
    qrnn = QRNN()
    hidden_out, c_out = qrnn(X, init_h, init_c, target=Y)
    print(hidden_out.size())
    
if __name__=="__main__":
    main()

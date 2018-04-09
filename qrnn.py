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
        
    def forward(self, Y, init_c):
        # Y = N * B * 3H
        # init_c = B * H
        n_seq = Y.size()[0]
        F, Z, O = Y.chunk(3, dim=2)

        # F, Z, O = N * B * H -> N * B * H
        # n_seq = N
        F = self.sigmoid(F)
        F = self.zoneout(F)
        Z = self.tanh(Z)
        O = self.sigmoid(O)

        # ZF : N * B * H
        ZF = torch.mul(1-F, Z)
        h_out = []
        c_out_list = [init_c]
        for t in range(n_seq):
            # c = B * H
            c = torch.add(torch.mul(F[t], c_out_list[-1]), ZF[t])
            # h = B * H
            h = torch.mul(c, O[t])
            h_out.append(h)
            c_out_list.append(c)

        # h_out = N * B * H
        # c_out = list(B * H), listlen=N+1
        # c_out[-1] = B * H
        h_out = torch.stack(h_out)
        return h_out, c_out_list[-1]

    def zoneout(self, F):
        F = 1 - nn.functional.dropout(1 - F, p=self.dropout_p)
        return F
            

        
class QRNN(nn.Module):

    def __init__(self, input_dim=64, hidden_size=64, dropout_p=0.2, kernel_size=1):
        super(QRNN, self).__init__()        
        self.conv1 = nn.Conv1d(input_dim, hidden_size*3, kernel_size) #B * C * L -> B * C_out * L_out
        self.fopooling = FOPooling(dropout_p=dropout_p)
        
    def forward(self, x, init_c):
        return self.conv2pool(x, init_c)

    def conv2pool(self, x, init_c):
        # x = N * B * H (or E)
        # init_h = B * H
        # init_c = B * H
        x = x.transpose(1, 2) #x = N * H * B
        x = self.conv1(x)     #x = N * 3H * B
        x = x.transpose(1, 2) #x = N * B * 3H
        # h_out = N * B * H
        # c_out = B * H
        x, c_out = self.fopooling(x,init_c)
        return x, c_out


class Attention(nn.Module):
    def __init__(self, hidden_size=64):
        super(Attention, self).__init__()
        self.softmax = nn.functional.softmax
        self.linear_score = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input_encode, target_encode, mask=None):
        #input_encode = N * B * H
        #target_encode = 1 * B * H
        #matrix = B * 1 * N
        #attn = B * 1 * N
        #mask = B * 1 * N

        #input_encode : N * B * H -> B * H * N
        input_encode = input_encode.transpose(0, 1).transpose(1, 2)

        #target_encode : 1 * B * H -> B * 1 * H
        target_encode = self.linear_score(target_encode)
        target_encode = target_encode.transpose(0, 1)
        
        #tmp = torch.cat([input_encode, target_encode], dim=2)
        #matrix : B * 1 * N
        matrix = torch.bmm(target_encode, input_encode)
        matrix -= matrix.mean(dim=2).unsqueeze(2)
        matrix = (matrix ** 2).sqrt()
        if mask is not None:
            #print(mask.size(), matrix.size())
            matrix.data.masked_fill_(mask, -float("Inf"))
            #print(matrix.data[0])

        #attn : B * 1 * N
        attn = self.softmax(matrix, dim=2)
        #print(attn.data[0])
        return attn

    def score(in_vect, tar_vect):
        pass

    
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

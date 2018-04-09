#coding:utf-8
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from qrnn import QRNN, Attention
USE_CUDA=True

class Decoder(nn.Module):
    def __init__(self, dict_size=60, hidden_size=64, embedding_size=32,
                 n_layers=2, dropout_p=0.2, kernel_size=1):
        
        super(Decoder, self).__init__()
        self.linear = nn.Linear(hidden_size, dict_size)
        self.softmax = nn.functional.softmax
        self.embed = nn.Embedding(dict_size, embedding_size,
                                   padding_idx=None)
        self.init_qrnn = QRNN(embedding_size, hidden_size, dropout_p, kernel_size)
        self.qrnn = QRNN(hidden_size, hidden_size, dropout_p, kernel_size)
        self.qrnn_list = [self.init_qrnn] + [self.qrnn] * (n_layers - 1)
        self.hidden_size = hidden_size
        self.attention = Attention(self.hidden_size)
        self.n_layers = n_layers
        self.attn_linear = nn.Linear(hidden_size*2, hidden_size)
        
    def forward(self, x, c_init_list, 
                encoder_out=None, mask=None):
        # M word から 次の 1 word の（条件付き)確率分布を生成する

        # x :1文字の入力ベクトル : 1 * B 
        # x : B -> 1 * B * E
        x = self.embed(x)
        c_out_list = []
        for n in range(self.n_layers):
            # x : 1 * B * E (or H) -> 1 * B * E
            # c_out -> B * H
            x, c_out = self.qrnn_list[n](x, c_init_list[n])
            c_out_list.append(c_out) 
        #このあと attention を加える
        if encoder_out is not None:
            # x : 1 * B * H
            # encoder_out : N * B * H
            # mask : B * 1 * N
            # attn : B * 1 * N
            attn = self.attention(encoder_out, x, mask)

            # encoder_out : N * B * H -> 1 * B * H
            encoder_out = torch.bmm(attn,
                                    encoder_out.transpose(0, 1)).transpose(0, 1)
            # x_cat : 1 * B * 2H
            x_cat = torch.cat([encoder_out, x], dim=2)

            # x (attn_vect) : 1 * B * H
            x = nn.functional.tanh(self.attn_linear(x_cat))
            
        #probs : 1 * B * DictSize
        probs = self.linear(x)
        #probs = self.softmax(probs, dim=1)

        # c_out_list : list(B * H), listlen=n_layers
        
        return probs, c_out_list, attn
        
        
class Encoder(nn.Module):
    def __init__(self, dict_size=60, hidden_size=64, embedding_size=32,
                 n_layers=2, dropout_p=0.2, kernel_size=1,
                 batch_size=50):
        super(Encoder, self).__init__()
        self.qrnn = QRNN(hidden_size, hidden_size, dropout_p, kernel_size)
        self.init_qrnn = QRNN(embedding_size, hidden_size, dropout_p, kernel_size)
        self.embed = nn.Embedding(dict_size, embedding_size,
                                   padding_idx=None)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.qrnn_list = [self.init_qrnn] + [self.qrnn] * (n_layers - 1)
        self.hidden_dims = [hidden_size] * n_layers 
        torch.manual_seed(1)
        if USE_CUDA:torch.cuda.manual_seed_all(1)
        
    def forward(self, x):
        # x : 入力ワードベクトル, : B * N (B : Batch, N : InputWordLength)
        # init_c : Qrnn の初期値(t = -1 の値) : B * E (n=0), B * H (n > 0) : n_layerl分
        batch_size = x.size()[0]
        init_c = [self.init_hidden(batch_size, dim) for dim in self.hidden_dims]

        # x : B * N -> N * B * E (E : Embedding dim)
        x = self.embed(x).transpose(0, 1)
        c_out_list = []

        for n in range(self.n_layers):
            # n = 0 : x : N*B*E, init_c : B*H
            # n > 0 : x : N*B*H, init_c : B*H
            x, c_out = self.qrnn_list[n](x, init_c[n])
            c_out_list.append(c_out)

        # c_out_list : list(B*H), listlen=n_layers
        return x, c_out_list

    def init_hidden(self, batch_size=50, dim=64):
        hidden = Variable(torch.rand(batch_size, dim) * 0.01) 
        if USE_CUDA: hidden = hidden.cuda()
        return hidden
    
            
if __name__=="__main__":
    #入力データは 20data * 50wordlen * 60dictsize * 64hiddensize    
    X = np.stack([np.arange(50) + 2 for _ in range(20)])
    Y = (X // 3) + 2
    X += np.random.randint(0, 3, X.shape)

    X = Variable(torch.from_numpy(X))
    Y = Variable(torch.from_numpy(Y))
    Sos = Variable(torch.zeros(20, 1).type(torch.LongTensor))
    print("X, Y size", X.size(), Y.size())

    encoder = Encoder()
    decoder = Decoder()

    encoder_out, encoder_c = encoder(X)
    print("len enc c", len(encoder_c))
    probs, c_out, attn = decoder(Sos,encoder_c, encoder_out=encoder_out)
    print(probs.size(), c_out[0].size())
    

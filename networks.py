#coding:utf-8
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from qrnn import QRNN, Attention
from torch.nn.functional import dropout

class Decoder(nn.Module):
    def __init__(self, dict_size=60, hidden_size=64, embedding=None, embedding_size=32,
                 n_layers=2, dropout_p=0.2, kernel_size=1, attention=True):
        
        super(Decoder, self).__init__()
        self.linear = nn.Linear(hidden_size, dict_size)
        self.softmax = nn.functional.softmax
        self.embed = embedding
        self.init_qrnn = QRNN(embedding_size, hidden_size, dropout_p, kernel_size)
        self.qrnn = QRNN(hidden_size, hidden_size, dropout_p, kernel_size)
        self.qrnn_list = [self.init_qrnn] + [self.qrnn] * (n_layers - 1)
        self.hidden_size = hidden_size
        self.attention = Attention(self.hidden_size)
        self.n_layers = n_layers
        self.attn_linear = nn.Linear(hidden_size*2, hidden_size)
        self.p = dropout_p
        self.use_attention = attention

        
    def forward(self, x, c_init_list, 
                encoder_out=None, mask=None):
        # M word から 次の 1 word の（条件付き)確率分布を生成する

        # x :1文字の入力ベクトル : 1 * B 
        # x : B -> 1 * B * E
        x = self.embed(x)

        # add dropout
        x = dropout(x, p=self.p, training=True)
        c_out_list = []
        for n in range(self.n_layers):
            # x : 1 * B * E (or H) -> 1 * B * E
            # c_out -> B * H
            x, c_out = self.qrnn_list[n](x, c_init_list[n])
            x = dropout(x, p=self.p, training=True)            
            c_out_list.append(c_out)
            
        #このあと attention を加える
        if (encoder_out is not None) and self.use_attention:
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
        else:
            attn = None
        #probs : 1 * B * DictSize
        probs = self.linear(x)
        #probs = self.softmax(probs, dim=1)

        # c_out_list : list(B * H), listlen=n_layers
        
        return probs, c_out_list, attn
        
        
class Encoder(nn.Module):
    def __init__(self, dict_size=60, hidden_size=64, embedding=None, embedding_size=32,
                 n_layers=2, dropout_p=0.2, kernel_size=1, use_cuda=False):
        super(Encoder, self).__init__()
        self.qrnn = QRNN(hidden_size, hidden_size, dropout_p, kernel_size)
        self.init_qrnn = QRNN(embedding_size, hidden_size, dropout_p, kernel_size)
        self.embed = embedding
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.qrnn_list = [self.init_qrnn] + [self.qrnn] * (n_layers - 1)
        self.hidden_dims = [hidden_size] * n_layers 
        torch.manual_seed(1)
        self.use_cuda = use_cuda
        if self.use_cuda:torch.cuda.manual_seed_all(1)
        self.p = dropout_p
        
    def forward(self, x):
        # x : 入力ワードベクトル, : B * N (B : Batch, N : InputWordLength)
        # init_c : Qrnn の初期値(t = -1 の値) : B * E (n=0), B * H (n > 0) : n_layerl分
        batch_size = x.size()[0]
        init_c = [self.init_hidden(batch_size, dim) for dim in self.hidden_dims]

        # x : B * N -> N * B * E (E : Embedding dim)
        x = self.embed(x).transpose(0, 1)

        # add dropout
        x = dropout(x, p=self.p)
        
        c_out_list = []

        for n in range(self.n_layers):
            # n = 0 : x : N*B*E, init_c : B*H
            # n > 0 : x : N*B*H, init_c : B*H
            x, c_out = self.qrnn_list[n](x, init_c[n])

            # add dropout
            x = dropout(x, p=self.p)

            c_out_list.append(c_out)

        # c_out_list : list(B*H), listlen=n_layers
        return x, c_out_list

    def init_hidden(self, batch_size=50, dim=64):
        hidden = Variable(torch.rand(batch_size, dim) * 0.001) 
        if self.use_cuda: hidden = hidden.cuda()
        return hidden


class LSTMDecoder(nn.Module):

    def __init__(self, dict_size=60, hidden_size=64, embedding=None, embedding_size=64,
                 n_layers=2, dropout_p=0.2, kernel_size=1, use_cuda=False, attention=True):
        
        super(LSTMDecoder, self).__init__()
        self.linear = nn.Linear(hidden_size, dict_size)
        self.softmax = nn.functional.softmax
        self.embed = embedding
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.p = dropout_p
        self.LSTM = nn.LSTM(input_size=embedding_size,
                            hidden_size=self.hidden_size, num_layers=self.n_layers,
                            dropout=self.p)
        self.use_cuda = use_cuda
        self.use_attention = attention
        if self.use_attention:
            self.attention = Attention(self.hidden_size)            
            self.attn_linear = nn.Linear(hidden_size*2, hidden_size)            
    def forward(self, x, encoder_c, encoder_out=None, mask=None):
        # M word から 次の 1 word の（条件付き)確率分布を生成する

        # x :1文字の入力ベクトル : 1 * B 
        # x : B -> 1 * B * E
        self.batch_size = x.size()[0]
        h_0, c_0 = encoder_c

        x = self.embed(x)

        # add dropout
        x = dropout(x, p=self.p, training=True)


        # x : 1 * B * E
        # h_0, c_0 : nlayer * B * H
        x, (h_t, c_t) = self.LSTM(x, (h_0, c_0))
            
        #このあと attention を加える
        if (encoder_out is not None) and self.use_attention:
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
        else:
            attn = None
        #probs : 1 * B * DictSize
        probs = self.linear(x)
        #probs = self.softmax(probs, dim=1)

        # c_out_list : list(B * H), listlen=n_layers
        
        return probs, (h_t, c_t), attn

    
class LSTMEncoder(nn.Module):
    def __init__(self, dict_size=60, hidden_size=64, embedding=None, embedding_size=64,
                 n_layers=2, dropout_p=0.2, kernel_size=1, use_cuda=False, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.embed = embedding
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.hidden_dims = [hidden_size] * n_layers 
        torch.manual_seed(1)
        self.use_cuda = use_cuda
        if self.use_cuda:torch.cuda.manual_seed_all(1)
        self.p = dropout_p
        self.LSTM = nn.LSTM(input_size=embedding_size, hidden_size=self.hidden_size, num_layers=self.n_layers, dropout=self.p, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        
    def forward(self, x):
        # x : 入力ワードベクトル, : B * N (B : Batch, N : InputWordLength)
        # init_c : Qrnn の初期値(t = -1 の値) : B * E (n=0), B * H (n > 0) : n_layerl分
        self.batch_size = x.size()[0]
        h_0, c_0 = self.init_hidden()

        # x : B * N -> N * B * E (E : Embedding dim)
        x = self.embed(x).transpose(0, 1)

        # add dropout
        x = dropout(x, p=self.p)
        x, (h_t, c_t) = self.LSTM(x, (h_0, c_0))        

        return x, (h_t, c_t)
    
    def init_hidden(self):
        bidirection = 2 if self.bidirectional else 1
        h_0 = torch.FloatTensor(self.n_layers*bidirection, self.batch_size, self.hidden_size)
        c_0 = torch.FloatTensor(self.n_layers*bidirection, self.batch_size, self.hidden_size)
        h_0.uniform_(-0.001, 0.001)
        c_0.uniform_(-0.001, 0.001)
        if self.use_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        return Variable(h_0), Variable(c_0)
    
            
if __name__=="__main__":
    #入力データは 20data * 50wordlen * 60dictsize * 64hiddensize    
    X = np.stack([np.arange(50) + 2 for _ in range(20)])
    Y = (X // 3) + 2
    X += np.random.randint(0, 3, X.shape)

    X = Variable(torch.from_numpy(X))
    Y = Variable(torch.from_numpy(Y))
    Sos = Variable(torch.zeros(20, 1).type(torch.LongTensor))
    print("X, Y size", X.size(), Y.size())
    emb = nn.Embedding(60, 64)
    encoder = LSTMEncoder(embedding=emb)
    decoder = LSTMDecoder(embedding=emb)

    z = Variable(torch.FloatTensor(2, 20, 64))
    encoder_out, encoder_c = encoder(X)
    print("len enc c", len(encoder_c))
    probs, c_out, attn = decoder(Sos,encoder_c, encoder_out=encoder_out)
    print(probs.size(), c_out[0].size())
    

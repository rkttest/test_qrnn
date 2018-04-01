#coding:utf-8
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from qrnn import QRNN, Attention
USE_CUDA=True
class Decoder(nn.Module):
    def __init__(self, dict_size=60, hidden_size=64,
                 n_layers=1, dropout_p=0.2, kernel_size=1):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(hidden_size, dict_size)
        self.softmax = nn.functional.softmax
        self.embed = nn.Embedding(dict_size, hidden_size,
                                   padding_idx=None)
        self.qrnn = QRNN(hidden_size,
                         n_layers, dropout_p, kernel_size)
        self.hidden_size = hidden_size
        self.attention = Attention()
        
    def forward(self, x, h_prev, c_prev, target=None,
                encoder_out=None):
        # M word から 次の 1 word の（条件付き)確率分布を生成する
        # target is not None -> attention
        x = self.embed(x).transpose(0, 1) #x = M * B * H
        h_out, c_out = self.qrnn(x, h_prev, c_prev)
        #このあと attention を加える
        if target is not None:
            target = self.embed(target).transpose(0, 1)
            target_h_out, target_c_out = self.qrnn(target, h_prev,
                                                   c_prev)
            attn = self.attention(encoder_out, target_h_out)
            encoder_out = torch.bmm(attn,
                                    encoder_out.transpose(0, 1)).transpose(0, 1)
            h_out += encoder_out

        h_out, c_out = self.qrnn(x, h_out[0], c_out[-1])
        probs = self.linear(h_out[-1])
        #probs = self.softmax(probs, dim=1)
        return probs, h_out[-1], c_out[-1], attn
        
        
class Encoder(nn.Module):
    def __init__(self, dict_size=60, hidden_size=64,
                 n_layers=2, dropout_p=0.2, kernel_size=1,
                 batch_size=50):
        super(Encoder, self).__init__()
        self.qrnn = QRNN(hidden_size,
                 n_layers, dropout_p, kernel_size)
        self.embed = nn.Embedding(dict_size, hidden_size,
                                   padding_idx=None)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        
    def forward(self, x, init_h, init_c):
	# 通常のSeq2Seq->decoderに1つ渡す
        # Attention->decoderに出力サイズのinputを渡す

        x = self.embed(x).transpose(0, 1) #X=B*N -> x=B*N*H
        h_out, c_out = self.qrnn(x, init_h, init_c)            
        return h_out, c_out

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.batch_size,
                                      self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden
    
            
if __name__=="__main__":
    #入力データは 20data * 50wordlen * 60dictsize * 64hiddensize    
    X = np.stack([np.arange(50) + 2 for _ in range(20)])
    Y = (X // 3) + 2
    X += np.random.randint(0, 3, X.shape)

    X = Variable(torch.from_numpy(X))
    Y = Variable(torch.from_numpy(Y[:,:1].reshape(-1, 1)))
    print("X, Y size", X.size(), Y.size())
    sos = Variable(torch.from_numpy(np.zeros((20), dtype=int).reshape(-1, 1)))

    init_h = Variable(torch.zeros(20, 64))
    init_c = Variable(torch.zeros(20, 64))

    encoder = Encoder()
    decoder = Decoder()
    attention = Attention()

    encoder_hidden, encoder_c = encoder(X, init_h, init_c, Y)
    decoder_init_h = encoder_hidden[-1]
    decoder_init_c = encoder_c[-1]
    probs, h_out, c_out = decoder(sos, decoder_init_h,
                                  decoder_init_c, encoder_out=encoder_hidden)

    print(probs.size(), h_out.size())
    

    

[]#coding:utf-8
import os, sys
from networks import Encoder, Decoder, LSTMEncoder, LSTMDecoder
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.nn.init import xavier_uniform, kaiming_uniform, xavier_normal
import torch.nn.functional as F
from optim.adam import Adam
import random
import numpy as np
import time

def xavier_init(model):
    count = 0
    for param in model.parameters():
        if len(param.size()) >= 2:
            count += 1
            xavier_normal(param)
            print(param.data.mean())

    print("--------total----------")
    print(count)



        
class EncoderDecoder(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=256,
                 n_layers=2, dropout_p=0.2, n_words=10000, max_word_len=50,
                 tokens=dict({"PAD":0, "SOS":1, "EOS":2, "UNK":3}),
                 use_cuda=False, attention=True):
        super(EncoderDecoder, self).__init__()
        self.embedding = nn.Embedding(n_words, embedding_size,
                                      padding_idx=None)

        self.encoder = Encoder(dict_size=n_words, embedding=self.embedding,
                               embedding_size=embedding_size,
                               hidden_size=hidden_size, n_layers=n_layers, use_cuda=use_cuda)
        self.decoder = Decoder(dict_size=n_words, embedding=self.embedding,
                               embedding_size=embedding_size,                               
                               hidden_size=hidden_size, n_layers=n_layers, attention=attention)
        self.use_cuda = use_cuda
        self.max_word_len = max_word_len
        self.tokens = tokens
        self.use_attention = attention
        
    def forward(self, x, target=None, getattention=False):
        batch_size = x.size()[0]
        encoder_out, encoder_c = self.encoder(x)
        decoder_input = Variable(torch.LongTensor([[self.tokens["SOS"]]*batch_size]))
        decoder_c = encoder_c
        mask = (x == 0).unsqueeze(1).data
        if self.use_cuda:
            decoder_input=decoder_input.cuda()
            mask = mask.cuda()
            
        decoder_outlist = []
        attention_out = []

        for di in range(self.max_word_len-1):
            pad_eos_num = (decoder_input.data == self.tokens["PAD"]).sum() +\
                          (decoder_input.data == self.tokens["EOS"]).sum()
            if pad_eos_num >= batch_size:
                break
            
            decoder_out, decoder_c, attention = self.decoder(decoder_input, decoder_c,
                                                             encoder_out=encoder_out,
                                                             mask=mask)
            topv, topi = decoder_out.data.topk(1)
            attention_out.append(attention)
            
            if target is None:
                decoder_input = Variable(topi[:,:,0]).detach()
                if self.use_cuda:decoder_input = decoder_input.cuda()
            else:
                decoder_input = target[:,di].unsqueeze(0).detach()

            decoder_outlist.append(decoder_out[0])
        decoder_outseq = torch.stack(decoder_outlist, dim=1) # B * M
        if self.use_cuda:decoder_outseq = decoder_outseq.cuda()

        if getattention:
            if self.use_attention:
                attention_out = torch.cat(attention_out, dim=1)
            else:
                attention_out = None
            return decoder_outseq, attention_out
        return decoder_outseq

    def word_forward(x, decoder_input, target=None, getattention=False):
        encoder_out, encoder_c = self.encoder(x)
        decoder_c = encoder_c
        mask = (x == 0).unsqueeze(1).data
        if self.use_cuda:
            decoder_input=decoder_input.cuda()
            mask = mask.cuda()
                        
        decoder_out, _, attention = self.decoder(decoder_input, decoder_c,
                                                 encoder_out=encoder_out,
                                                 mask=mask)
        return decoder_out, attention

class LSTMEncoderDecoder(EncoderDecoder):
    def __init__(self, embedding_size=128, hidden_size=256,
                 n_layers=2, dropout_p=0.2, n_words=10000, max_word_len=50,
                 tokens=dict({"PAD":0, "SOS":1, "EOS":2, "UNK":3}),
                 use_cuda=False, attention=True):
        super(LSTMEncoderDecoder, self).__init__()
        self.embedding = nn.Embedding(n_words, embedding_size,
                                      padding_idx=None)
        
        self.encoder = LSTMEncoder(dict_size=n_words, embedding=self.embedding,
                               embedding_size=embedding_size,
                               hidden_size=hidden_size, n_layers=n_layers, use_cuda=use_cuda)
        self.decoder = LSTMDecoder(dict_size=n_words, embedding=self.embedding,
                               embedding_size=embedding_size,
                                   hidden_size=hidden_size, n_layers=n_layers, use_cuda=use_cuda, attention=attention)
        self.max_word_len = max_word_len
        self.tokens = tokens
        self.use_attention = attention
    
class Trainer(object):

    def __init__(self, model, optimizer, lossfn, trainloader=None,
                 valloader=None, testloader=None, save_dir="./", clip_norm=4.,
                 teacher_forcing_ratio=0.5, epoch=10, save_freq=1000, dictionary=None, target_dist=None):
        self.model = model
        self.optim = optimizer
        self.lossfn = lossfn
        self.save_dir = save_dir
        self.clip_norm = clip_norm
        self.epoch = epoch
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.save_freq = save_freq
        self.epoch = epoch
        self.n_iter = 0
        self.dictionary = dictionary
        self.target_dist = None
        np.random.seed(1)
        torch.manual_seed(1)
        if self.model.use_cuda:
            torch.cuda.manual_seed_all(1)
            self.model = self.model.cuda()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    def model_initialize(self, model_path=None):
        if model_path is None:
            xavier_init(self.model)
        else:
            param = torch.load(model_path)
            self.model.load_state_dict(param)
        
    def train(self, summary_writer=None):
        for epoch in range(self.epoch):
            print("epoch {}".format(epoch))
            loss_list = []
            start = time.time()
            for idx, tensors in enumerate(self.trainloader):
                x = Variable(tensors[0])
                target = Variable(tensors[1])
                if self.model.use_cuda:
                    x = x.cuda()
                    target = target.cuda()
                    
                if np.random.rand() < self.teacher_forcing_ratio:
                    model_out = self.model.forward(x, target=target[:,1:])
                else:
                    model_out = self.model.forward(x)
                loss, size = self.get_loss(model_out, target[:,1:])
                loss_list.append(loss.data[0] / size)                
                self.optimize(loss)

                if (idx+1) % self.save_freq == 0:
                    print("batch idx", idx)
                    end = time.time()
                    print("time : ", end -start )
                    start = end
                    print("Varidation Start")
                    val_loss, attention = self.validation()
                    self.save_model(os.path.join(self.save_dir,
                                                 "epoch{}_batchidx{}".format(epoch, idx)))
                    print("train loss {}".format(np.mean(loss_list)))
                    print("val loss {}".format(val_loss))
                    if summary_writer is not None:
                        self.summary_write(summary_writer, np.mean(loss_list), val_loss,
                                           attention)

                    loss_list = []
                self.n_iter += 1
            if summary_writer is not None:
                summary_writer.export_scalars_to_json(os.path.join(self.save_dir, "all_scalars_{}.json".format(epoch)))

                
    def summary_write(self, writer, train_loss, val_loss, attention=None):
        writer.add_scalar("data/train_loss", train_loss , self.n_iter)
        writer.add_scalar("data/val_loss", val_loss , self.n_iter)
        if (attention is not None) and self.use_attention:
            writer.add_image("data/attention", attention[0,:,-25:], self.n_iter)
        
    def validation(self):
        losses = []
        for idx, tensors in enumerate(self.valloader):
            x = Variable(tensors[0])
            target = Variable(tensors[1])
            if self.model.use_cuda:
                x = x.cuda()
                target = target.cuda()
            
            model_out, attention = self.model.forward(x, getattention=True)
            probs = nn.functional.softmax(model_out, dim=2)
            if self.target_dist is not None:
                target_dist = self.target_dist.type(type(probs.data))
                topk, topi = (probs.data - target_dist).topk(3)
            else:
                topk, topi = probs.data.topk(3)

            if self.dictionary is not None:
                model_seq = topi.cpu().numpy()[:4,:,0]
                in_seq = x.data.cpu().numpy()[:4]
                target_seq = target.data.cpu().numpy()[:4]
                batchsize = model_seq.shape[0]
                
                for batch in range(batchsize):
                    input_sentence = self.dictionary.indexlist2sentence(in_seq[batch][::-1])
                    target_sentence = self.dictionary.indexlist2sentence(target_seq[batch])
                    model_sentence = self.dictionary.indexlist2sentence(model_seq[batch])
            
                    print("-----------------------------------------")
                    print("> ", input_sentence)
                    print("= ", target_sentence)
                    print("< ", model_sentence)
                    print("-----------------------------------------")
                    print(" ")
                    
                
            loss, size = self.get_loss(model_out, target)
            losses.append(loss.data[0] / size)
        return np.mean(losses), attention

    
    def test(self, target_dist=None):
        test_out = []
        
        for idx, tensors in enumerate(self.testloader):
            x = Variable(tensors[0])
            target = Variable(tensors[1])
            if self.model.use_cuda:
                x = x.cuda()
                target = target.cuda()
            
            model_out = self.model.forward(x)
            probs = nn.functional.softmax(model_out, dim=2)
            if target_dist is not None:
                target_dist = target_dist.type(type(probs.data))
                topk, topi = (probs.data - target_dist).topk(3)
            else:
                topk, topi = probs.data.topk(3)
            
            test_out.append([topi.cpu().numpy()[:,:,0], x.data.cpu().numpy(),
                            target.data.cpu().numpy()])
            
        return test_out

    def optimize(self, loss):
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip_norm)        
        self.optim.step()

    def get_loss(self, predict, target):
        length = min(predict.size()[1], target.size()[1])
        # weight = (target != self.model.tokens["PAD"]).data
        # weight = weight.cpu().numpy() #weight = B * M
        loss = 0
        for di in range(length):
            # index = np.where(weight[:, di])[0]
            # if index.shape[0] == 0:
            #     break
            # index = torch.from_numpy(index)
            # if self.model.use_cuda: index = index.cuda()
            # loss += self.lossfn(predict[index][:,di], target[index][:,di])
            loss += self.lossfn(predict[:,di], target[:,di])

        return loss, (di + 1)
    
    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

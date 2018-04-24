#coding:utf-8
import os, sys
from networks import Encoder, Decoder, LSTMEncoder, LSTMDecoder, GRUEncoder, GRUDecoder
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.nn.init import xavier_uniform, kaiming_uniform, xavier_normal
import torch.nn.functional as F
from optim.adam import Adam
import random

from seq2seq.models.DecoderRNN import DecoderRNN
from seq2seq.models.EncoderRNN import EncoderRNN
from seq2seq.models.TopKDecoder import TopKDecoder


import numpy as np
np.random.seed(1)
import time

def xavier_init(model):
    count = 0
    for param in model.parameters():
        if len(param.size()) >= 2:
            count += 1
            kaiming_uniform(param)
            print(param.data.abs().mean())

    print("--------total----------")
    print(count)

        
class EncoderDecoder(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=256,
                 n_layers=2, dropout_p=0.2, n_words=10000, max_word_len=50,
                 tokens=dict({"PAD":0, "SOS":1, "EOS":2, "UNK":3}),
                 use_cuda=False, attention=True, bidirectional=False, target_dist=None):
        super(EncoderDecoder, self).__init__()
        self.embedding = nn.Embedding(n_words, embedding_size,
                                      padding_idx=None)
        self.n_layers = n_layers
        self.encoder = Encoder(dict_size=n_words, embedding=self.embedding,
                               embedding_size=embedding_size,
                               hidden_size=hidden_size, n_layers=n_layers-1,
                               use_cuda=use_cuda)
        self.decoder = Decoder(dict_size=n_words, embedding=self.embedding,
                               embedding_size=embedding_size,                               
                               hidden_size=hidden_size,
                               n_layers=n_layers, attention=attention)
        self.use_cuda = use_cuda
        self.max_word_len = max_word_len
        self.tokens = tokens
        self.use_attention = attention
        self.bidirectional = bidirectional
        self.target_dist = target_dist
        self.multihead_attention = False
        
    def forward(self, x, target=None, getattention=False):
        batch_size = x.size()[0]
        encoder_out, encoder_c = self.encoder(x)
        
        init_h = self.encoder.init_hidden()
        init_h = init_h[:self.n_layers]
        decoder_c = torch.cat([encoder_c[-1].unsqueeze(0), init_h])
        if target is None:
            decoder_input = Variable(torch.LongTensor([[self.tokens["SOS"]]*batch_size]))
        else:
            decoder_input = target[:,0].unsqueeze(0)

        if type(decoder_c) == tuple:
            decoder_c = list(encoder_c)
            decoder_c[1] = Variable(torch.zeros(encoder_c[1].size()))
            if self.use_cuda:
                decoder_c[1] = decoder_c[1].cuda()
        mask = (x == 0).unsqueeze(1).data
        if self.use_cuda:
            decoder_input = decoder_input.cuda()
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
            if self.target_dist is not None:
                decoder_out.data -= self.target_dist

            topv, topi = decoder_out.data.topk(1)
            attention_out.append(attention)
            
            if target is None:
                decoder_input = Variable(topi[:,:,0]).detach()
                if self.use_cuda:decoder_input = decoder_input.cuda()
            else:
                decoder_input = target[:,di+1].unsqueeze(0).detach()

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


class BeamEncoderDecoder(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=256,
                 n_layers=2, dropout_p=0.2, n_words=10000, max_word_len=50,
                 tokens=dict({"PAD":0, "SOS":1, "EOS":2, "UNK":3}),
                 use_cuda=False, attention=True, bidirectional=False,
                 beam_search=True, topk=3):
        super(BeamEncoderDecoder, self).__init__()
        self.topk = topk
        self.embedding = nn.Embedding(n_words, hidden_size,
                                      padding_idx=None)
        self.encoder = EncoderRNN(n_words, max_word_len, hidden_size, n_layers=n_layers,
                                  dropout_p=dropout_p, input_dropout_p=dropout_p)
        self.decoder = DecoderRNN(n_words, max_word_len-1, hidden_size, n_layers=n_layers,
                                  sos_id=tokens["SOS"], eos_id=tokens["EOS"],
                                  dropout_p=dropout_p, input_dropout_p=dropout_p,
                                  use_attention=attention)
        self.topk_decoder = TopKDecoder(self.decoder, self.topk)
        self.encoder.embedding = self.embedding        

        self.decoder.embedding = self.embedding
        self.use_cuda = use_cuda
        self.use_attention = attention
        
    def forward(self, x, target=None, getattention=False):
        encoder_out, encoder_c = self.encoder(x)
        if target is None:
            decoder_outputs, decoder_c, attn = self.topk_decoder(encoder_outputs=encoder_out,
                                                           encoder_hidden=encoder_c)
        else:
            decoder_outputs, decoder_c, attn = self.decoder(inputs=target,
                                                         encoder_outputs=encoder_out,
                                                         encoder_hidden=encoder_c)
            
        if getattention:
            if self.use_attention:
                print(attn.keys())
                attention_out = torch.stack(attn[DecoderRNN.KEY_ATTN_SCORE][0])[:,0,0]
            else:
                attention_out = None
            return torch.stack(decoder_outputs,dim=1), attention_out
        return torch.stack(decoder_outputs, dim=1)
   
        
class LSTMEncoderDecoder(EncoderDecoder):
    def __init__(self, embedding_size=128, hidden_size=256,
                 n_layers=2, dropout_p=0.2, n_words=10000, max_word_len=50,
                 tokens=dict({"PAD":0, "SOS":1, "EOS":2, "UNK":3}),
                 use_cuda=False, attention=True, bidirectional=False,
                 residual=True, blockterm=True):
        super(LSTMEncoderDecoder, self).__init__()
        self.embedding = nn.Embedding(n_words, embedding_size,
                                      padding_idx=None)
        
        self.encoder = LSTMEncoder(dict_size=n_words, embedding=self.embedding,
                               embedding_size=embedding_size,
                                   hidden_size=hidden_size, n_layers=n_layers,
                                   use_cuda=use_cuda, bidirectional=bidirectional,
                                   residual=residual, blockterm=blockterm)
        self.decoder = LSTMDecoder(dict_size=n_words, embedding=self.embedding,
                               embedding_size=embedding_size,
                                   hidden_size=hidden_size, n_layers=n_layers,
                                   use_cuda=use_cuda, attention=attention,
                                   residual=residual, blockterm=blockterm)
        self.max_word_len = max_word_len
        self.tokens = tokens
        self.use_attention = attention
        self.bidirectional = bidirectional

class GRUEncoderDecoder(EncoderDecoder):
    def __init__(self, embedding_size=128, hidden_size=256,
                 n_layers=2, dropout_p=0.2, n_words=10000, max_word_len=50,
                 tokens=dict({"PAD":0, "SOS":1, "EOS":2, "UNK":3}),
                 use_cuda=False, attention=True, bidirectional=False, residual=False,
                 target_dist=None, outdict_size=10000):
        super(GRUEncoderDecoder, self).__init__()
        #self.embedding = nn.Embedding(n_words, embedding_size,
        #                              padding_idx=None)
        self.embedding = None
        self.n_layers = n_layers        
        self.encoder = GRUEncoder(dict_size=n_words, embedding=self.embedding,
                               embedding_size=embedding_size,
                                   hidden_size=hidden_size, n_layers=n_layers,
                                  use_cuda=use_cuda, bidirectional=bidirectional,
                                  residual=residual, n_words=n_words, dropout_p=dropout_p)
        self.decoder = GRUDecoder(dict_size=n_words, embedding=self.embedding,
                               embedding_size=embedding_size,
                                   hidden_size=hidden_size,
                                  n_layers=n_layers,
                                   use_cuda=use_cuda,
                                  attention=attention,
                                  residual=residual,
                                  outdict_size=outdict_size, dropout_p=dropout_p)
        self.max_word_len = max_word_len
        self.tokens = tokens
        self.use_attention = attention
        self.bidirectional = bidirectional
        self.use_cuda = use_cuda
        self.target_dist = target_dist
        
class Trainer(object):
    
    def __init__(self, model, optimizer, lossfn, trainloader=None,
                 valloader=None, testloader=None, save_dir="./", clip_norm=4.,
                 teacher_forcing_ratio=0.5, epoch=10, save_freq=1000,
                 dictionary=None, target_dist=None, scheduler=None, 
                 getattention=False, beam_search=False, use_mixer=False):

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
        self.scheduler = scheduler
        self.beam_search = beam_search
        self.getattention = getattention
        self.use_mixer = use_mixer
        self.rl_alpha = 10
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
            perplexity_list = []
            start = time.time()
            if self.scheduler is not None:
                self.scheduler.step()
            for idx, tensors in enumerate(self.trainloader):
                if type(self.scheduler) == CycleLR:
                    self.scheduler.step()
                
                if (idx+1) % 200 == 0:
                    print(idx+1)
                    print(np.mean(loss_list))
                x = Variable(tensors[0])
                target = Variable(tensors[1])
                if self.model.use_cuda:
                    x = x.cuda()
                    target = target.cuda()

                if np.random.random() < self.teacher_forcing_ratio:
                    model_out = self.model.forward(x, target) 
                else:
                    model_out = self.model.forward(x)    
                loss, size = self.get_loss(model_out, target[:,1:])
                loss_list.append(loss.data[0] / size)                
                self.optimize(loss)
                perplexity = self.get_perplexity(model_out, target[:,1:])
                perplexity_list.append(perplexity)                
                if (idx+1) % self.save_freq == 0:
                    print("batch idx", idx)
                    end = time.time()
                    print("time : ", end -start )
                    start = end
                    print("Varidation Start")
                    val_loss, attention = self.validation()
                    self.save_model(os.path.join(self.save_dir,
                                                 "epoch{}_batchidx{}".format(epoch, idx+1)))
                    print("train loss {}".format(np.mean(loss_list)))
                    print("val loss {}".format(val_loss))
                    print("train perplexity {}".format(np.mean(perplexity_list)))
                    
                    if summary_writer is not None:
                        self.summary_write(summary_writer, np.mean(loss_list), val_loss,
                                           attention)

                    loss_list = []
                    perplexity_list = []
                self.n_iter += 1
            if summary_writer is not None:
                summary_writer.export_scalars_to_json(os.path.join(self.save_dir, "all_scalars_{}.json".format(epoch)))

                
    def summary_write(self, writer, train_loss, val_loss, attention=None):
        writer.add_scalar("data/train_loss", train_loss , self.n_iter)
        writer.add_scalar("data/val_loss", val_loss , self.n_iter)
        if (attention is not None):
            writer.add_image("data/attention", attention[0,:,-25:], self.n_iter)
        
    def validation(self):
        losses = []
        for idx, tensors in enumerate(self.valloader):
            x = Variable(tensors[0])
            target = Variable(tensors[1])
            if self.model.use_cuda:
                x = x.cuda()
                target = target.cuda()

            if self.getattention:
                model_out, attention = self.model.forward(x, getattention=self.getattention)
            else:
                model_out = self.model.forward(x, getattention=self.getattention)
                attention = None
            probs = nn.functional.softmax(model_out, dim=2)
            if self.target_dist is not None:
                target_dist = self.target_dist.type(type(probs.data))
                topk, topi = (probs.data - target_dist).topk(3)
                
            else:
                topk, topi = probs.data.topk(3)
            loss, size = self.get_loss(model_out, target[:,1:])
            losses.append(loss.data[0] / size)

            if (self.dictionary is not None) and (idx == 0):
                model_seq = topi.cpu().numpy()[:15,:,0]
                in_seq = x.data.cpu().numpy()[:15]
                target_seq = target.data.cpu().numpy()[:15]
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
        loss = 0
        for di in range(length):
            loss += self.lossfn(predict[:,di], target[:,di])


        if self.use_mixer:
            probs = nn.functional.softmax(predict, dim=1)            
            reward = self.bleu(probs, target)
            reward = torch.from_numpy(reward).unsqueeze(1)
            if self.use_cuda: reward = reward.cuda()
            obj = (probs*reward).view(-1, probs.size[-1])
            target = target.contiguous()
            obj = obj[target.view(-1)]
            loss += self.rl_alpha * obj

        return loss, (di + 1)

    def get_perplexity(self, predict, target):
        
        probs = nn.functional.softmax(predict, dim=1).detach()
        probs = probs.view(-1, probs.size()[-1]).contiguous()
        target = target.contiguous()
        target = target.view(-1)
        perplexity = (1. / (probs[:,target] + 1e-10)).mean()
        return perplexity.data[0]
    
    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    @staticmethod
    def bleu(model_pred, target, N=4):
        """
        BLEUの計算を行う
        model_pred:B*M
        target:B*N
        return B*N
        """
        denom = 0
        numer = 0
        batch_size = target.size()[0]
        max_target_len = target.size()[1]
        r = np.zeros(batch_size)
        c = np.zeros(batch_size)
        p_n = np.zeros(batch_size, N)
        for batch in range(batch_size):
            for n in range(1, N+1):
                model_len = (model[batch] > len(self.tokens.keys())).sum()
                pred_n_gram = [model[batch][i:i+n] for i in range(model_len-n+1)]
                target_len = (target[batch] > len(self.tokens.keys())).sum()
                target_n_gram = [target[batch][i:i+n] for i in range(target_len-n+1)]
                denom = len(target_n_gram)
                numer = len([_ for i in range(target_len) if ((pred_n_gram[i]==target_n_gram[i]).data.sum()==n)])
                r[batch] += target_len
                c[batch] += model_len
                p_n[batch, n-1] = denom / numer
        BP = np.minimum(1, np.exp(1 - c / r))
        BLEU = BP * np.sum(np.log(p_n+1e-10), axis=1)
        return BLEU


from torch.optim.lr_scheduler import _LRScheduler
class CycleLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, min_lr=1e-5,
                 max_lr=1e-3, cycle_step=4000):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.cycle_step = cycle_step
        super(CycleLR, self).__init__(optimizer, last_epoch=-1)                
        #base_lr = min_lr
        
    def get_lr(self):
        return [self.calc_lr(base_lr) for base_lr in self.base_lrs]


    def calc_lr(self, base_lr):
        step = self.last_epoch % self.cycle_step
        c = base_lr + 2 * (self.max_lr - base_lr) * ((step + 1) / self.cycle_step)
        lr = c - 2 * max(0, c - self.max_lr)

        if (step + 1) == self.cycle_step:
            self.max_lr *= 0.95
            self.max_lr = max(self.max_lr, base_lr+5e-5)
        return lr


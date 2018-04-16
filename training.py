#coding:utf-8

import os, shutil, sys
import numpy as np
np.random.seed(1)
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import DataLoader
from optim.adam import Adam
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
from model import EncoderDecoder, Trainer, LSTMEncoderDecoder, GRUEncoderDecoder
from tensorboardX import SummaryWriter

from hyperparam import HP

sys.path.append("../json")
from dictionary import WordDict


def main():
    wd = WordDict()
    wd.load_dict(pd.read_csv("../json/w2i.csv"))
    
    s2s_model = LSTMEncoderDecoder(embedding_size=HP.embedding_size,
                               hidden_size=HP.hidden_size,
                               n_layers=HP.n_layers,
                               dropout_p=HP.dropout_p,
                               n_words=HP.n_words,
                               max_word_len=HP.max_word_len,
                               tokens=HP.tokens,
                                  use_cuda=HP.USE_CUDA, attention=HP.use_attention)

    print("Model", s2s_model)

    #weight = torch.FloatTensor(np.load("../json/weight.npy"))
    lossfn = nn.CrossEntropyLoss(ignore_index=HP.tokens["PAD"])#, weight=weight)
    optimizer = Adam(s2s_model.parameters(),
                     lr=HP.learning_rate, amsgrad=True, weight_decay=HP.l2)

    with open("../json/textlist.pkl", "rb") as f:
        import pickle
        textdata = pickle.load(f)
    train_data = textdata[:40]#(len(textdata)//10)*8]
    val_data = train_data#textdata[(len(textdata)//10)*8:]
    val_data = val_data[:400]

    np.random.shuffle(train_data)

    print("train_size", len(train_data))
    print("val_size", len(val_data))
    trainloader = DataLoader(train_data, batch_size=HP.batch_size, collate_fn=collate_fn)
    valloader = DataLoader(val_data, batch_size=HP.batch_size, collate_fn=collate_fn)
    
    trainer = Trainer(model=s2s_model, optimizer=optimizer, lossfn=lossfn,
                      trainloader=trainloader, epoch=HP.epoch,
                      valloader=valloader, save_dir=HP.save_dir, save_freq=HP.save_freq,
                      dictionary=wd, teacher_forcing_ratio=0.5)
    
    #shutil.copy("hyperparam.py", os.path.join(HP.save_dir, "hyperparam.py"))    
    #writer = SummaryWriter()
    trainer.model_initialize()#os.path.join(HP.save_dir,"epoch17_batchidx199"))
    #
    trainer.train()#writer)


def collate_fn(sample):
    src = [s[0] for s in sample]
    tar = [s[1] for s in sample]

    max_src_len = max([len(s) for s in src])
    max_tar_len = max([len(s) for s in tar])
    PAD = 0
    src = [s + [PAD] * (max_src_len - len(s)) for s in src]
    tar = [s + [PAD] * (max_tar_len - len(s)) for s in tar]

    src = [s[::-1] for s in src]
    
    src = torch.LongTensor(src)
    tar = torch.LongTensor(tar)
    use_cuda = False
    if use_cuda:
        src = src.cuda()
        tar = tar.cuda()

    return (src, tar)
        
        
if __name__=="__main__":
    main()

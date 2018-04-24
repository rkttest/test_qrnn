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
from torch.utils.data.sampler import WeightedRandomSampler, Sampler
from optim.lr_scheduler import MultiStepLR
from model import CycleLR
from model import EncoderDecoder, Trainer, LSTMEncoderDecoder, GRUEncoderDecoder, BeamEncoderDecoder
from tensorboardX import SummaryWriter

from hyperparam import HP
sys.path.append("../../src")
sys.path.append("../json")
from wordsdictionary import simpleWordDict, ssWordDict
#from dictionary import WordDict

import random
random.seed(1)
INDEX=""
def main():
    wd = ssWordDict("../../Dictionary/newdata/{}WordDict.csv".format(INDEX),
                    "../../Dictionary/newdata/{}TypeDict.csv".format(INDEX))
    #wd = WordDict()
    #wd.load_dict(pd.read_csv("../json/w2i.csv"))
    target_dist = np.load("../../Dictionary/newdata/{}target_dist.npy".format(INDEX))
    target_dist = np.r_[target_dist, np.zeros(HP.outdict_size - target_dist.shape[0])]
    target_dist = np.log(target_dist + 1e-8)
    
    target_dist = torch.from_numpy(target_dist)
    target_dist = target_dist.type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)

    if HP.USE_CUDA: target_dist = target_dist.cuda()
    s2s_model = GRUEncoderDecoder(embedding_size=HP.embedding_size,
                               hidden_size=HP.hidden_size,
                                  outdict_size=HP.outdict_size,
                               n_layers=HP.n_layers,
                               dropout_p=HP.dropout_p,
                               n_words=HP.n_words,
                               max_word_len=HP.max_word_len,
                               tokens=HP.tokens,
                               use_cuda=HP.USE_CUDA,
                                attention=HP.use_attention, bidirectional=True,
                                   residual=True, target_dist=target_dist)


    # wd = ssWordDict("../../Dictionary/WordDict.csv", "../../Dictionary/TypeDict.csv")
    # s2s_model = GRUEncoderDecoder(embedding_size=HP.embedding_size,
    
    #weight = torch.FloatTensor(np.load("../../TrainData/dict_weight.npy"))

    #if HP.USE_CUDA: weight = weight.cuda()
    #lossfn = nn.CrossEntropyLoss(weight=weight, ignore_index=HP.tokens["PAD"])
    lossfn = nn.CrossEntropyLoss(ignore_index=HP.tokens["PAD"])

    # optimizer = torch.optim.SGD(s2s_model.parameters(), lr=HP.learning_rate, momentum=0.9,
    #                 weight_decay=HP.l2, nesterov=True)
    # scheduler = CycleLR(optimizer, max_lr=0.01, cycle_step=4000) 
    # optimizer = Adam(s2s_model.parameters(),
    #lr=HP.learning_rate, amsgrad=True, weight_decay=HP.l2)
    # optimizer = Adam(s2s_model.embedding.parameters(),                     
    #                   lr=HP.learning_rate, amsgrad=True, weight_decay=HP.l2)
    
    optimizer = torch.optim.Adam(s2s_model.parameters(),
                     lr=HP.learning_rate, weight_decay=HP.l2)
    scheduler = CycleLR(optimizer, max_lr=0.002, cycle_step=4000) 
    #scheduler = MultiStepLR(optimizer, milestones=[10,  20], gamma=0.4)

    # train_arr = np.load("../../TrainData/corpus_train_merged.npy")
    # train_arr = train_arr.reshape(-1, 2, train_arr.shape[1])[:,:,:HP.max_word_len+1]
    # train_arr[:,0,:] = train_arr[:,0,::-1]

    # train_arr = torch.from_numpy(train_arr)
    # weight = np.ones(train_arr.size()[0])
    # weight = weight / weight.sum()
    # val_arr = np.load("../../TrainData/corpus_val_merged.npy")
    # val_arr = val_arr.reshape(-1, 2, val_arr.shape[1])[:1000,:,:HP.max_word_len+1]
    # val_arr[:,0,:] = val_arr[:,0,::-1]
    # val_arr = torch.from_numpy(val_arr)

    # train_data = TensorDataset(train_arr[:,0,:-1], train_arr[:,1])
    
    # val_data = TensorDataset(val_arr[:,0,:-1], val_arr[:,1])
    # trainloader = DataLoader(train_data, batch_size=HP.batch_size,
    #                          sampler=WeightedRandomSampler(weight,
    #                                                        num_samples=train_arr.size()[0]))
    # valloader = DataLoader(val_data, batch_size=HP.batch_size)
    print("Model", s2s_model)

    #weight = torch.FloatTensor(np.load("../json/weight.npy"))
    with open("../../Dictionary/newdata/{}filteredlist_ag.pkl".format(INDEX), "rb") as f:
        #with open("../json/textlist.pkl", "rb") as f:        
        import pickle
        textdata = pickle.load(f)

    train_data = textdata[:(len(textdata)//10)*9]
    random.shuffle(train_data)
    train_data = sorted(train_data, key=lambda x:len(x[0]))
    #sampling_weight = np.load("../../Dictionary/newdata/sampling_weight.npy")
    sampling_weight = np.ones(len(train_data))
    sampling_weight = sampling_weight[:(len(textdata)//10)*8]
    sampling_weight = sampling_weight / sampling_weight.sum()
    val_data = textdata[(len(textdata)//10)*9:]
    random.shuffle(val_data)
    val_data = val_data[:4000]
    val_data = sorted(val_data, key=lambda x:len(x[0]))

    #np.random.shuffle(train_data)

    print("train_size", len(train_data))
    print("val_size", len(val_data))
    trainloader = DataLoader(train_data, batch_size=HP.batch_size, collate_fn=collate_fn,
                             sampler=RandomBatchSampler(train_data, HP.batch_size))
                             # sampler=WeightedRandomSampler(sampling_weight[:-HP.batch_size],
                             #                                   num_samples=len(train_data)))
    valloader = DataLoader(val_data, batch_size=HP.batch_size, collate_fn=collate_fn)
    
    trainer = Trainer(model=s2s_model, optimizer=optimizer, lossfn=lossfn,
                      trainloader=trainloader, epoch=HP.epoch,
                      valloader=valloader, save_dir=HP.save_dir, save_freq=HP.save_freq,
                      dictionary=wd, teacher_forcing_ratio=0.2, scheduler=scheduler,
                      beam_search=False, getattention=True)

    shutil.copy("hyperparam.py", os.path.join(HP.save_dir, "hyperparam.py"))    
    trainer.model_initialize("SavedModel/34/epoch0_batchidx4000")
    
    writer = SummaryWriter()
    trainer.train(writer)
    trainer.train()
    

class RandomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        np.random.seed(1)
        
    def __iter__(self):
        #batchサイズ分連続した値を出す
        mat = np.arange(len(self.data_source) - len(self.data_source) % self.batch_size).reshape(-1, self.batch_size)
        np.random.shuffle(mat)
        vect = mat.reshape(-1)
        return iter(vect)
    
def collate_fn(sample):
    PAD = HP.tokens["PAD"]
    SOS = HP.tokens["SOS"]

    src = [s[0] for s in sample]
    tar = [s[1] for s in sample]

    max_src_len = max([len(s) for s in src])
    max_tar_len = max([len(s) for s in tar])
    #min_src_len = min([len(s) for s in src])
    #min_tar_len = min([len(s) for s in tar])
    #print("src", max_src_len, min_src_len)
    
    
    # src = [s[0] if s[1][0] == SOS else [s[1][0]] + s[0] for s in sample]
    # tar = [[SOS] + s[1][1:] for s in sample]

    # max_src_len = max([len(s) for s in src])
    # max_tar_len = max([len(s) for s in tar])
    src = [s + [PAD] * (max_src_len - len(s)) for s in src]
    tar = [s + [PAD] * (max_tar_len - len(s)) for s in tar]

    src = [s[::-1] for s in src]
    src = torch.from_numpy(np.array(src))
    tar = torch.from_numpy(np.array(tar))
    use_cuda = HP.USE_CUDA
    
    if use_cuda:
        src = src.cuda()
        tar = tar.cuda()

    return (src, tar)
        
        
if __name__=="__main__":
    main()

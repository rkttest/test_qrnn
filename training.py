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
from optim.lr_scheduler import MultiStepLR
from model import EncoderDecoder, Trainer, LSTMEncoderDecoder, GRUEncoderDecoder
from tensorboardX import SummaryWriter

from hyperparam import HP
sys.path.append("../../src")
from wordsdictionary import simpleWordDict, ssWordDict



def main():
    #wd = simpleWordDict("../../Dictionary/datum/reshape_merged_dict.csv")    
    wd = ssWordDict("../../Dictionary/WordDict.csv", "../../Dictionary/TypeDict.csv")
    s2s_model = GRUEncoderDecoder(embedding_size=HP.embedding_size,
                               hidden_size=HP.hidden_size,
                               n_layers=HP.n_layers,
                               dropout_p=HP.dropout_p,
                               n_words=HP.n_words,
                               max_word_len=HP.max_word_len,
                               tokens=HP.tokens,
                               use_cuda=HP.USE_CUDA,
                               attention=HP.use_attention)
    
    #weight = torch.FloatTensor(np.load("../../TrainData/dict_weight.npy"))
    sampling_weight = np.load("../../Dictionary/sampling_weight.npy")
    #if HP.USE_CUDA: weight = weight.cuda()
    #lossfn = nn.CrossEntropyLoss(weight=weight, ignore_index=HP.tokens["PAD"])
    lossfn = nn.CrossEntropyLoss(ignore_index=HP.tokens["PAD"])

    optimizer = Adam(s2s_model.parameters(),
                     lr=HP.learning_rate, amsgrad=True, weight_decay=HP.l2)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)
    
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
    with open("../../Dictionary/filteredlist.pkl", "rb") as f:
        import pickle
        textdata = pickle.load(f)
    train_data = textdata[:(len(textdata)//10)*8]
    sampling_weight = sampling_weight[:(len(textdata)//10)*8]
    sampling_weight = sampling_weight / sampling_weight.sum()
    val_data = textdata[(len(textdata)//10)*8:]
    val_data = val_data[:1000]

    #np.random.shuffle(train_data)

    print("train_size", len(train_data))
    print("val_size", len(val_data))
    trainloader = DataLoader(train_data, batch_size=HP.batch_size, collate_fn=collate_fn,
                             sampler=WeightedRandomSampler(sampling_weight,
                                                               num_samples=len(train_data)))
    valloader = DataLoader(val_data, batch_size=HP.batch_size, collate_fn=collate_fn)
    
    trainer = Trainer(model=s2s_model, optimizer=optimizer, lossfn=lossfn,
                      trainloader=trainloader, epoch=HP.epoch,
                      valloader=valloader, save_dir=HP.save_dir, save_freq=HP.save_freq,
                      dictionary=wd, scheduler=scheduler)

    shutil.copy("hyperparam.py", os.path.join(HP.save_dir, "hyperparam.py"))    

    writer = SummaryWriter()
    trainer.model_initialize("SavedModel/16/epoch18_batchidx4000")
    trainer.train(writer)


def collate_fn(sample):
    src = [s[0] for s in sample]
    tar = [s[1] for s in sample]

    max_src_len = max([len(s) for s in src])
    max_tar_len = max([len(s) for s in tar])
    PAD = HP.tokens["PAD"]
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

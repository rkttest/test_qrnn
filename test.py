#coding:utf-8
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from optim.adam import Adam
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
from model import EncoderDecoder, Trainer
from tensorboardX import SummaryWriter

from hyperparam import HP

sys.path.append("../json")
from dictionary import WordDict

def main():
    s2s_model = EncoderDecoder(embedding_size=HP.embedding_size,
                               hidden_size=HP.hidden_size,
                               n_layers=HP.n_layers,
                               dropout_p=HP.dropout_p,
                               n_words=HP.n_words,
                               max_word_len=HP.max_word_len,
                               tokens=HP.tokens,
                               use_cuda=HP.USE_CUDA)
    
    lossfn = nn.CrossEntropyLoss()
    optimizer = Adam(s2s_model.parameters(),
                     lr=HP.learning_rate, amsgrad=True, weight_decay=HP.l2)
    target_dist = np.zeros(HP.n_words)
    _target_dist = np.load("../json/target_dist.npy")
    target_dist[:_target_dist.shape[0]] = _target_dist * 1.2
    target_dist = target_dist.reshape(1, 1, -1)
    target_dist = torch.from_numpy(target_dist).type(torch.DoubleTensor)
    
    test_arr = np.load("../json/train.npy")
    test_arr = test_arr.reshape(-1, 2, test_arr.shape[1])[:400,:,:HP.max_word_len]
    test_arr[:,0,:] = test_arr[:,0,::-1]
    test_arr = torch.from_numpy(test_arr)
    
    test_data = TensorDataset(test_arr[:,0], test_arr[:,1])
    testloader = DataLoader(test_data, batch_size=HP.batch_size)
    trainer = Trainer(model=s2s_model, optimizer=optimizer, lossfn=lossfn,
                      testloader=testloader, epoch=HP.epoch,
                      save_dir="SavedModel/19", save_freq=HP.save_freq)

    model_path = "SavedModel/19/epoch18_batchidx199"
    trainer.model_initialize(model_path)
    test_out = trainer.test()#target_dist)

    wd = WordDict()
    wd.load_dict(pd.read_csv("../json/w2i.csv"))
    
    for outseq in test_out:
        model_out = outseq[0]
        inseq = outseq[1]
        target = outseq[2]
        batchsize = model_out.shape[0]
        
        for batch in range(batchsize):
            input_sentence = wd.indexlist2sentence(inseq[batch][::-1])
            target_sentence = wd.indexlist2sentence(target[batch])
            model_sentence = wd.indexlist2sentence(model_out[batch])
            
            print("-----------------------------------------")
            print("> ", input_sentence)
            print("= ", target_sentence)
            print("< ", model_sentence)
            print("-----------------------------------------")
            print(" ")
if __name__=="__main__":
    main()

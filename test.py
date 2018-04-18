#coding:utf-8
import os, sys
sys.path.append("../../src")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from optim.adam import Adam
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
from model import EncoderDecoder, Trainer, LSTMEncoderDecoder, GRUEncoderDecoder
from tensorboardX import SummaryWriter

from hyperparam import HP

from wordsdictionary import simpleWordDict

def main():
    s2s_model = GRUEncoderDecoder(embedding_size=HP.embedding_size,
                               hidden_size=HP.hidden_size,
                               n_layers=HP.n_layers,
                               dropout_p=HP.dropout_p,
                               n_words=HP.n_words,
                               max_word_len=HP.max_word_len,
                               tokens=HP.tokens,
                               use_cuda=HP.USE_CUDA)


    wd = simpleWordDict("../../Dictionary/datum/reshape_merged_dict.csv")

    lossfn = nn.CrossEntropyLoss()
    optimizer = Adam(s2s_model.parameters(),
                     lr=HP.learning_rate, amsgrad=True, weight_decay=HP.l2)

    test_arr = np.load("../../TrainData/corpus_test_merged.npy")

    # target_dist = np.zeros(HP.n_words)
    # _target_dist = np.load("../json/target_dist.npy")
    # target_dist[:_target_dist.shape[0]] = _target_dist * 1.2
    # target_dist = target_dist.reshape(1, 1, -1)
    # target_dist = torch.from_numpy(target_dist).type(torch.DoubleTensor)
    
    # test_arr = np.load("../json/train.npy")

    test_arr = test_arr.reshape(-1, 2, test_arr.shape[1])[:400,:,:HP.max_word_len]
    test_arr[:,0,:] = test_arr[:,0,::-1]
    test_arr = torch.from_numpy(test_arr)
    
    test_data = TensorDataset(test_arr[:,0], test_arr[:,1])
    testloader = DataLoader(test_data, batch_size=HP.batch_size)
    trainer = Trainer(model=s2s_model, optimizer=optimizer, lossfn=lossfn,
                      testloader=testloader, epoch=HP.epoch,
                      save_dir="SavedModel/11", save_freq=HP.save_freq)

    model_path = "SavedModel/15/epoch9_batchidx2000"

    trainer.model_initialize(model_path)
    test_out = trainer.test()#target_dist)

    #wd = WordDict()
    #wd.load_dict(pd.read_csv("../json/w2i.csv"))
    
    for outseq in test_out:
        model_out = outseq[0]
        inseq = outseq[1]
        target = outseq[2]
        batchsize = model_out.shape[0]
        
        for batch in range(batchsize):
            input_sentence = wd.indexlist2sentence(inseq[batch][::-1])
            input_sentence = "".join([w for w in input_sentence if w not in ["EOS", "PAD", "SOS"]])
            target_sentence = wd.indexlist2sentence(target[batch])
            target_sentence = "".join([w for w in target_sentence if w not in ["EOS", "PAD", "SOS"]])            
            model_sentence = wd.indexlist2sentence(model_out[batch])
            model_sentence = "".join([w for w in model_sentence if w not in ["EOS", "PAD", "SOS"]])
                                     
            print("-----------------------------------------")
            print("> ", input_sentence)
            print("= ", target_sentence)
            print("< ", model_sentence)
            print("-----------------------------------------")
            print(" ")
if __name__=="__main__":
    main()

#coding:utf-8

import os, shutil
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from optim.adam import Adam
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
from model import EncoderDecoder, Trainer
from tensorboardX import SummaryWriter

from hyperparam import HP

def main():
    s2s_model = EncoderDecoder(embedding_size=HP.embedding_size,
                               hidden_size=HP.hidden_size,
                               n_layers=HP.n_layers,
                               dropout_p=HP.dropout_p,
                               n_words=HP.n_words,
                               max_word_len=HP.max_word_len,
                               tokens=HP.tokens,
                               use_cuda=HP.USE_CUDA)
    weight = torch.FloatTensor(np.load("../../TrainData/dict_weight.npy"))
    if HP.USE_CUDA: weight = weight.cuda()
    lossfn = nn.CrossEntropyLoss(weight=weight, ignore_index=HP.tokens["PAD"])
    optimizer = Adam(s2s_model.parameters(),
                     lr=HP.learning_rate, amsgrad=True, weight_decay=HP.l2)
    
    train_arr = np.load("../../TrainData/corpus_train_merged.npy")
    train_arr = train_arr.reshape(-1, 2, train_arr.shape[1])[:,:,:HP.max_word_len+1]
    train_arr[:,0,:] = train_arr[:,0,::-1]

    train_arr = torch.from_numpy(train_arr)
    weight = np.ones(train_arr.size()[0])
    weight = weight / weight.sum()
    val_arr = np.load("../../TrainData/corpus_val_merged.npy")
    val_arr = val_arr.reshape(-1, 2, val_arr.shape[1])[:600,:,:HP.max_word_len+1]
    val_arr[:,0,:] = val_arr[:,0,::-1]
    val_arr = torch.from_numpy(val_arr)

    train_data = TensorDataset(train_arr[:,0,:-1], train_arr[:,1,1:])
    
    val_data = TensorDataset(val_arr[:,0,:-1], val_arr[:,1,1:])
    trainloader = DataLoader(train_data, batch_size=HP.batch_size,
                             sampler=WeightedRandomSampler(weight, num_samples=train_arr.size()[0]))
    valloader = DataLoader(val_data, batch_size=HP.batch_size)

    trainer = Trainer(model=s2s_model, optimizer=optimizer, lossfn=lossfn,
                      trainloader=trainloader, epoch=HP.epoch,
                      valloader=valloader, save_dir=HP.save_dir, save_freq=HP.save_freq)

    shutil.copy("hyperparam.py", os.path.join(HP.save_dir, "hyperparam.py"))    
    writer = SummaryWriter()
    trainer.model_initialize()
    #
    trainer.train(writer)

    
if __name__=="__main__":
    main()

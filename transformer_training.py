#coding:utf-8

import os, shutil, sys
sys.path.append("../seq2seq.pytorch/")
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from optim.adam import Adam
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
#from model import EncoderDecoder, Trainer
from tensorboardX import SummaryWriter

from hyperparam import HP

from seq2seq.models.transformer import Transformer
from seq2seq.tools.trainer import Seq2SeqTrainer


def main():
    # s2s_model = EncoderDecoder(embedding_size=HP.embedding_size,
    #                            hidden_size=HP.hidden_size,
    #                            n_layers=HP.n_layers,
    #                            dropout_p=HP.dropout_p,
    #                            n_words=HP.n_words,
    #                            max_word_len=HP.max_word_len,
    #                            tokens=HP.tokens,
    #                            use_cuda=HP.USE_CUDA)
    s2s_model = Transformer(HP.n_words, hidden_size=64, num_heads=4, inner_linear=128, num_layers=3)
    weight = torch.FloatTensor(np.load("../json/weight.npy"))
    lossfn = nn.CrossEntropyLoss(ignore_index=HP.tokens["PAD"], weight=weight)
    optimizer = Adam(s2s_model.parameters(),
                     lr=HP.learning_rate, amsgrad=True, weight_decay=HP.l2)
    
    train_arr = np.load("../json/train.npy")
    train_arr = train_arr.reshape(-1, 2, train_arr.shape[1])[:,:,:HP.max_word_len]
    train_len = (train_arr > 0).sum(axis=2)    
    train_arr[:,0,:] = train_arr[:,0,::-1]
    train_arr = torch.from_numpy(train_arr)

    weight = np.ones(train_arr.size()[0])
    weight = weight / weight.sum()
    val_arr = np.load("../json/val.npy")
    val_arr = val_arr.reshape(-1, 2, val_arr.shape[1])[:600,:,:HP.max_word_len]
    val_len = (val_arr > 0).sum(axis=2)
    val_arr[:,0,:] = val_arr[:,0,::-1]
    val_arr = torch.from_numpy(val_arr)

    #train_data = TensorDataset(train_arr[:,0], train_arr[:,1])
    #val_data = TensorDataset(val_arr[:,0], val_arr[:,1])
    
    # trainloader = DataLoader(train_data, batch_size=HP.batch_size,
    #                          sampler=WeightedRandomSampler(weight, num_samples=train_arr.size()[0]))
    # valloader = DataLoader(val_data, batch_size=HP.batch_size)

    train_data = list(zip(list(zip(train_arr[:,0], train_len[:,0])),
                          list(zip(train_arr[:,1], train_len[:,1]))))
    print(len(train_data), len(train_data[0]))
    trainloader = DataLoader(train_data,
                             batch_size=HP.batch_size,
                             sampler=WeightedRandomSampler(weight, num_samples=train_arr.size()[0]))
    valloader = DataLoader(list(zip(list(zip(val_arr[:,0], val_len[:,0])),
                                    list(zip(val_arr[:,1], val_len[:,1])))),
                           batch_size=HP.batch_size)

    # trainer = Trainer(model=s2s_model, optimizer=optimizer, lossfn=lossfn,
    #                   trainloader=trainloader, epoch=HP.epoch,
    #                   valloader=valloader, save_dir=HP.save_dir, save_freq=HP.save_freq)

    trainer = Seq2SeqTrainer(s2s_model, cuda=False, save_path=HP.save_dir)
    trainer.optimizer.optimizer = optimizer
    shutil.copy("hyperparam.py", os.path.join(HP.save_dir, "hyperparam.py"))    
    #writer = SummaryWriter()
    #trainer.model_initialize()#os.path.join(HP.save_dir,"epoch15_batchidx199"))
    #
    #trainer.train(writer)

    while trainer.epoch < HP.epoch:
        print("epoch", int(trainer.epoch))
        trainer.run(trainloader, valloader)


if __name__=="__main__":
    main()

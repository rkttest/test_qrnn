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

from wordsdictionary import simpleWordDict, ssWordDict
INDEX=""
def main():
    wd = ssWordDict("../../Dictionary/newdata/WordDict.csv",
                    "../../Dictionary/newdata/TypeDict.csv")
    #wd = simpleWordDict("../../Dictionary/datum/reshape_merged_dict.csv")

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

    
    lossfn = nn.CrossEntropyLoss()
    optimizer = Adam(s2s_model.parameters(),
                     lr=HP.learning_rate, amsgrad=True, weight_decay=HP.l2)

    #test_arr = np.load("../../TrainData/corpus_test_merged.npy")

    # target_dist = np.zeros(HP.n_words)
    # _target_dist = np.load("../json/target_dist.npy")
    # target_dist[:_target_dist.shape[0]] = _target_dist * 1.2
    # target_dist = target_dist.reshape(1, 1, -1)
    # target_dist = torch.from_numpy(target_dist).type(torch.DoubleTensor)
    
    # test_arr = np.load("../json/train.npy")
    with open("../../Dictionary/newdata/filteredlist.pkl", "rb") as f:
        import pickle
        textdata = pickle.load(f)
    test_data = textdata[:(len(textdata)//10)*9]
    test_data = test_data[4000:4500]
    checks = [
        "おはよう!",
        "お話ししようよ",
        "今日は暑いね",
        "調子はどう?"
    ]
    checks = list(map(lambda x:(wd.sentence2indexlist(x) + [2], [1]), checks))
    test_data += checks
    test_data = sorted(test_data, key=lambda x:len(x[0]))
    
    # test_arr = test_arr.reshape(-1, 2, test_arr.shape[1])[:400,:,:HP.max_word_len]
    # test_arr[:,0,:] = test_arr[:,0,::-1]
    # test_arr = torch.from_numpy(test_arr)
    
    # test_data = TensorDataset(test_arr[:,0], test_arr[:,1])
    testloader = DataLoader(test_data,
                            batch_size=HP.batch_size, collate_fn=collate_fn)
    trainer = Trainer(model=s2s_model,
                      optimizer=optimizer,
                      lossfn=lossfn,
                      testloader=testloader,
                      epoch=HP.epoch,
                      save_dir="SavedModel/11", save_freq=HP.save_freq)

    model_path = "SavedModel/34/epoch16_batchidx4000"
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

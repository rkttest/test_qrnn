#coding:utf-8
import os, sys
sys.path.append("../../src")
from loaddata import Datum
from wordsdictionary import WordsDictionary
from networks import Encoder, Decoder, USE_CUDA
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
PAD_token = 0
SOS_token = 1
EOS_token = 2
from train import train
DEBUG = False

def main():
    ### データの読み込み
    train_path = "../../TrainData/corpus_val_merged.npy"
    train_data = np.load(train_path)
    train_pair = train_data.reshape(-1, 2, train_data.shape[1])
    train_size = train_pair.shape[0]
    np.random.seed(1)
    shuffle_arr = np.arange(train_size)
    np.random.shuffle(shuffle_arr)
    train_pair = train_pair[shuffle_arr]
    train_pair = train_pair[:400]
    train_size = train_pair.shape[0]
    ### ここまでを別の処理で行う

    ### ハイパーパラメータの定義
    print("train size", train_pair.shape)
    attn_model = 'general'
    embedding_size = 256
    hidden_size = 512
    n_layers = 2
    dropout_p = 0
    n_words = 52000
    batch_size = 4
    plot_every = 10
    print_every = 1
    learning_rate = 1e-4
    
    ### ネットワークの定義
    encoder = Encoder(dict_size=n_words,
                      embedding_size=embedding_size,
                         hidden_size=hidden_size,
                         n_layers=n_layers, batch_size=batch_size)                         
    decoder = Decoder(dict_size=n_words,
                      embedding_size=embedding_size,
                      hidden_size=hidden_size,
                      n_layers=n_layers)

    
    # encoder_param = torch.load("SavedModel/1/encoder_5_0")
    # decoder_param = torch.load("SavedModel/1/decoder_5_0")
    # encoder.load_state_dict(encoder_param)
    # decoder.load_state_dict(decoder_param)

    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()
    
    criterion = nn.NLLLoss()

    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every
    start = time.time()

    mean_loss_list = []
    index_list = []
    ###

    init_epoch = 1
    n_epochs = 10
    n_batch = 5
    base_dir = "SavedModel/9/"
    
    for epoch in range(init_epoch, n_epochs+1):

        ### バッチデータごとに処理
        for batch in range(n_batch):
            print("fname", base_dir + "encoder_{}_{}".format(epoch, batch*2000))
            total_loss = 0
            
            encoder_param = torch.load(base_dir + "encoder_{}_{}".format(epoch, batch*2000))
            decoder_param = torch.load(base_dir + "decoder_{}_{}".format(epoch, batch*2000))
            encoder.load_state_dict(encoder_param)
            decoder.load_state_dict(decoder_param)

            for batch_idx in range(train_size // batch_size):
                if DEBUG: print(batch_idx)
                training_pair = train_pair[batch_idx*batch_size:(batch_idx+1)*batch_size]
                input_variable = Variable(torch.LongTensor(training_pair[:,0]))
                target_variable = Variable(torch.LongTensor(training_pair[:,1]))
                
                if USE_CUDA:
                    input_variable = input_variable.cuda()
                    target_variable = target_variable.cuda()
                
                # Run the train function
                loss = train(input_variable, target_variable, encoder, decoder, teacher_forcing_ratio=1)
                total_loss += loss
            mean_loss = total_loss / (train_size // batch_size)
            mean_loss_list.append(mean_loss)
            index_list.append((epoch, batch))
            print("mean loss :", mean_loss)
        
    mean_loss_arr = np.array(mean_loss_list)


    plt.plot((np.arange(len(mean_loss_list))/n_batch)[1:], mean_loss_arr[1:])
    plt.savefig("nmll_graph.png")
    plt.close()

    print("minimul loss:", np.min(mean_loss_arr))
    print("minimul model:{}epoch + {}data".format(*index_list[np.argmin(mean_loss_arr)]))

def train(input_variable, target_variable, encoder, decoder, max_length=50, teacher_forcing_ratio=0.5):

    clip = 4.0
    
    # Zero gradients of both optimizers
    loss = 0 # Added onto for each word
    
    # Get size of input and target sentences
    batch_size = input_variable.size()[0]     # input_variable = B * N 
    input_length = input_variable.size()[1]   
    target_length = target_variable.size()[1] # target_variable = B * M
    
    # Run words through encoder
    encoder_out, encoder_c = encoder(input_variable)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]*batch_size]))
    decoder_c = encoder_c
    weight = (target_variable != PAD_token).data
    weight = weight.cpu().numpy() # weight = B * M

    # Generate attention mask
    mask = torch.stack([input_variable == 0 for _ in range(target_length)], dim=1).data


    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        mask = mask.cuda()
        
    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio

    decoder_outlist = [torch.LongTensor([SOS_token]*batch_size)]
    if USE_CUDA:decoder_outlist = [torch.cuda.LongTensor([[SOS_token]]*batch_size)]

    for di in range(target_length-1):
        if DEBUG: print("di", di)
        decoder_output, decoder_c, attn = decoder(decoder_input, decoder_c,
                                       encoder_out=encoder_out,
                                       mask=mask[:,di,:].unsqueeze(1))

        index = np.where(weight[:, di+1])[0]
        if len(index) == 0:
            break
        index = torch.from_numpy(index)
        if USE_CUDA: index = index.cuda()
        loss += nn.CrossEntropyLoss()(decoder_output[0][index],target_variable[index][:, di+1])
        topv, topi = decoder_output.data.topk(1)
        decoder_outlist.append(topi)

        if use_teacher_forcing:
            decoder_input = target_variable[:,di+1].unsqueeze(0).detach() # Next target is next input
        else:
            #print(decoder_outlist)
            if USE_CUDA:
                decoder_input = Variable(torch.cuda.LongTensor(topi[:,:,0])).detach()
            else:
                decoder_input = Variable(torch.LongTensor(topi[:,:,0])).detach()
                #print("decoder size", decoder_input.size())
            
    
    return loss.data[0] / max(1, di) #target_length

if __name__ == "__main__":
    main()


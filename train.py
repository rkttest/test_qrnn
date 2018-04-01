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
PAD_token = 0
SOS_token = 1
EOS_token = 2

DEBUG = False

def main():
    ### データの読み込み
    train_path = "../../TrainData/train.npy"
    train_data = np.load(train_path)
    train_pair = train_data.reshape(-1, 2, train_data.shape[1])
    train_size = train_pair.shape[0]
    np.random.seed(1)
    shuffle_arr = np.arange(train_size)
    np.random.shuffle(shuffle_arr)
    train_pair = train_pair[shuffle_arr]
    ### ここまでを別の処理で行う

    ### ハイパーパラメータの定義
    print("train size", train_pair.shape)
    attn_model = 'general'
    hidden_size = 512
    n_layers = 2
    dropout_p = 0.2
    n_words = 55000
    batch_size = 20
    n_epochs = 1000
    plot_every = 10
    print_every = 1
    learning_rate = 1e-4
    
    ### ネットワークの定義
    encoder = Encoder(dict_size=n_words,
                         hidden_size=hidden_size,
                         n_layers=n_layers, batch_size=batch_size)                         
    decoder = Decoder(dict_size=n_words,
                      hidden_size=hidden_size,
                      n_layers=n_layers)
    if not os.path.exists("SavedModel"):
        os.mkdir("SavedModel")
    # encoder_param = torch.load("SavedModel/1/encoder_5_0")
    # decoder_param = torch.load("SavedModel/1/decoder_5_0")
    # encoder.load_state_dict(encoder_param)
    # decoder.load_state_dict(decoder_param)
    init_epoch = 1
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()
    encoder_optimizer = optim.Adam(encoder.parameters(),
                                   lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(),
                                   lr=learning_rate)
    
    criterion = nn.NLLLoss()

    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every
    start = time.time()

    ### 学習開始
    for epoch in range(init_epoch, n_epochs + 1):
        print("epoch :", epoch)
        ### バッチデータごとに処理
        for batch_idx in range(train_size // batch_size):
            if DEBUG: print(batch_idx)
            training_pair = train_pair[batch_idx*batch_size:(batch_idx+1)*batch_size]
            input_variable = Variable(torch.LongTensor(training_pair[:,0]))
            target_variable = Variable(torch.LongTensor(training_pair[:,1]))

            if USE_CUDA:
                input_variable = input_variable.cuda()
                target_variable = target_variable.cuda()
                
            # Run the train function
            loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            if DEBUG:
                print("loss :", loss)
                print("time :", time.time()-start)
            # Keep track of loss
            print_loss_total += loss
            plot_loss_total += loss
            
            if (batch_idx + 1)% print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = plot_loss_total
                end = time.time()
                print("batch idx : {}/{}".format(batch_idx, train_size // batch_size))
                print("time :", end-start)
                start = end
                print("loss avg :{}\nloss total : {}".format(print_loss_avg, print_loss_total))
                print_loss_total = 0
        
            if (epoch % 1 == 0) and (batch_idx % 1000 == 0):
                torch.save(encoder.state_dict(), "SavedModel/encoder_{}_{}".format(epoch, batch_idx))
                torch.save(decoder.state_dict(), "SavedModel/decoder_{}_{}".format(epoch, batch_idx))


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=50):
    teacher_forcing_ratio = 10.5
    clip = 4.0
    
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word
    
    # Get size of input and target sentences
    batch_size = input_variable.size()[0]     # input_variable = B * N 
    input_length = input_variable.size()[1]   
    target_length = target_variable.size()[1] # target_variable = B * M

    
    # Run words through encoder
    encoder_hidden = encoder.init_hidden() # encoder_hidden = B * H
    encoder_c = encoder.init_hidden()      # encoder_c = B * H
    encoder_hidden, encoder_c = encoder(input_variable, encoder_hidden, encoder_c)
    # encoder_hidden = N * B * H (1dim is list), encoder_c = N * B * H
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]]*batch_size))
    decoder_c = encoder_c[-1]
    decoder_hidden = encoder_hidden[-1] # Use last hidden state from encoder to start decoder

    weight = (target_variable != PAD_token).data
    weight = weight.cpu().numpy() # weight = B * M

    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_c = decoder_c.cuda()
        
        # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if DEBUG: print("esize", encoder_hidden.size())

    #if use_teacher_forcing:
        # Teacher forcing: Use the ground-truth target as the next input
        # decoder input = target variable
        
    for di in range(target_length-1):
        if DEBUG: print("di", di)
        decoder_output, _, _, _ = decoder(decoder_input, 
                                          decoder_hidden,
                                          decoder_c,
                                          target=target_variable[:,:di+1],
                                          encoder_out=encoder_hidden)
        #decoder_output = B * Dict
        index = np.where(weight[:, di+1])[0]
        if len(index) == 0:
            break
        index = torch.from_numpy(index)
        if USE_CUDA: index = index.cuda()
        loss += nn.CrossEntropyLoss()(decoder_output[index],target_variable[index][:, di+1])

        decoder_input = target_variable[:,:di+2] # Next target is next input

    # if use_teacher_forcing:
    #     pass
    # else:
    #     # Without teacher forcing: use network's own prediction as the next input
    #     for di in range(target_length-1):
    #         if DEBUG: print("di", di)

    #         decoder_output, _, _, _ = decoder(decoder_input, 
    #                                           decoder_hidden,
    #                                           decoder_c,
    #                                           target=target_variable[:,:di+1],
    #                                           encoder_out=encoder_hidden)
    #         index = np.where(weight[di+1])[0]
    #         if len(index) == 0:
    #             break
    #         index = torch.from_numpy(index)
    #         if USE_CUDA: index = index.cuda()
    #         loss += nn.CrossEntropyLoss()(decoder_output[index],
    #                           target_variable[di+1][index])

    #         # Get most likely word index (highest value) from output
    #         topv, topi = decoder_output.data.topk(1)
    #         decoder_input = Variable(topi) #.unsqueeze(1) #Variable(torch.LongTensor([[ni]])) # Chosen word is next input
    #         if USE_CUDA: decoder_input = decoder_input.cuda()
            
    #         # Stop at end of sentence (not necessary when using known targets)
    #         if (topi == EOS_token).max(): break
            
    # Backpropagation
    if DEBUG: print("Start BackWard")
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    if DEBUG: print("Start Optimize")
    encoder_optimizer.step()
    decoder_optimizer.step()

    if DEBUG: print("Done")
    
    return loss.data[0] / target_length


if __name__ == "__main__":
    main()


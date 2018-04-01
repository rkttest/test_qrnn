#coding:utf-8
import os, sys
sys.path.append("../../src")
from loaddata import Datum
from wordsdictionary import WordsDictionary, simpleWordDict
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import random

from train import PAD_token, SOS_token, EOS_token
from networks import Encoder, Decoder, USE_CUDA

MAX_LENGTH = 50
DEBUG=False
def main():
    test_path = "../../TrainData/sample.npy"
    test_data = np.load(test_path)
    test_pairs = test_data.reshape(-1, 2, test_data.shape[1])
    dictpath = "../../Dictionary/datum/reshape_dict.csv"
    dictionary = simpleWordDict(dictpath)
    
    attn_model = 'general'
    hidden_size = 512
    n_layers = 2
    dropout_p = 0.1
    n_words = 55000
    
    en_path = "SavedModel/encoder_8_0"
    de_path = "SavedModel/decoder_8_0"
    encoder_param = torch.load(en_path)
    decoder_param = torch.load(de_path)

    ### ネットワークの定義
    encoder = Encoder(dict_size=n_words,
                         hidden_size=hidden_size,
                         n_layers=n_layers, batch_size=1)
    decoder = Decoder(dict_size=n_words,
                      hidden_size=hidden_size,
                      n_layers=n_layers)

    encoder.load_state_dict(encoder_param)
    decoder.load_state_dict(decoder_param)

    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()
        
    test_num = 20
    random.seed(0)
    for i in range(test_num):
        test_pair = random.choice(test_pairs)
        input_variable = Variable(torch.LongTensor(test_pair[0].reshape(1, -1)))
       
        output_words = evaluate(dictionary,
                                input_variable,
                                encoder, decoder)
        output_words = [w for w in output_words]
        output_sentence = ''.join(output_words)

        input_sentence = dictionary.indexlist2sentence(test_pair[0])
        input_sentence = "".join([i for i in input_sentence if i not in ["PAD", "EOS"]])
        target_sentence = dictionary.indexlist2sentence(test_pair[1])
        target_sentence = "".join([i for i in target_sentence if i not in ["PAD", "EOS", "SOS"]])
        print('>', input_sentence)
        print('=', target_sentence)
        print('<', output_sentence)
        print('')

def evaluate(dictionary, input_variable, encoder, decoder,
             max_length=MAX_LENGTH):

    input_variable = input_variable[:, :MAX_LENGTH]
    input_length = input_variable.size()[0]
    if USE_CUDA:
        input_variable = input_variable.cuda()
        
    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_c = encoder.init_hidden()
    encoder_hidden, encoder_c = encoder(input_variable, encoder_hidden, encoder_c)

    # Create starting vectors for decoder
    decoder_c = encoder_c[-1]
    decoder_hidden = encoder_hidden[-1] # Use 
        
    decoded_words = []
    decoder_outs = [SOS_token]
    decoder_attentions = torch.zeros(max_length, max_length)

    # Run through decoder
    for di in range(max_length-1):
        decoder_input = Variable(torch.LongTensor([decoder_outs]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
        
        if DEBUG:
            print(di)
            print("decoder_hidden ", decoder_hidden.size())
            print(decoder_input.size())

        decoder_output, decoder_hidden, decoder_c, decoder_attention = decoder(decoder_input, decoder_hidden, decoder_c, target=decoder_input, encoder_out=encoder_hidden)
        
        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        decoder_hidden = decoder_hidden[0, -1]
        decoder_c = decoder_c[-1]
        if DEBUG: print(topi.size(), topi)
        ni = topi[0][0]
        if ni == EOS_token or ni == PAD_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoder_outs.append(ni)
            decoded_words.append(dictionary.index2word(ni))
            
    
    return decoded_words #, decoder_attentions[:di+1, :len(encoder_outputs)]
  

if __name__=="__main__":
    main()

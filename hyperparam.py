#coding:utf-8

class HyperParams(object):
    def __init__(self):
        pass


HP = HyperParams()


HP.embedding_size = 200 #300
HP.hidden_size = 200
HP.n_layers = 3
HP.dropout_p = 0.2
HP.n_words = 26000 #26500 #27000 #33500
HP.outdict_size = 12500 #21100 #21500 #27000
HP.max_word_len = 25
HP.tokens = dict({"PAD":0, "SOS":1, "EOS":2, "UNK":3})

HP.USE_CUDA = False
HP.learning_rate = 0.00005
HP.l2 = 2e-6
HP.batch_size = 100
HP.epoch = 300
HP.save_freq = 4000
HP.save_dir = "SavedModel/34"
HP.use_attention = True

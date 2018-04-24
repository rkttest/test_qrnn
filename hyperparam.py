#coding:utf-8

class HyperParams(object):
    def __init__(self):
        pass


HP = HyperParams()


HP.embedding_size = 100 #300
HP.hidden_size = 100
HP.n_layers = 2
HP.dropout_p = 0.2
HP.n_words = 7000
HP.outdict_size = 7000
HP.max_word_len = 25
HP.tokens = dict({"PAD":0, "SOS":1, "EOS":2, "UNK":3})

HP.USE_CUDA = False
HP.learning_rate = 0.0005
HP.l2 = 2e-6
HP.batch_size = 300
HP.epoch = 300
HP.save_freq = 40
HP.save_dir = "SavedModel/35"
HP.use_attention = True

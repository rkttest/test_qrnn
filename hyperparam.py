#coding:utf-8

class HyperParams(object):
    def __init__(self):
        pass


HP = HyperParams()

HP.embedding_size = 200
HP.hidden_size = 400
HP.n_layers = 2
HP.dropout_p = 0.2
HP.n_words = 52000
HP.max_word_len = 50
HP.tokens = dict({"PAD":0, "SOS":1, "EOS":2, "UNK":3})
HP.USE_CUDA = True
HP.learning_rate = 0.0005
HP.l2 = 0.00005
HP.batch_size = 2
HP.epoch = 10
HP.save_freq = 4000

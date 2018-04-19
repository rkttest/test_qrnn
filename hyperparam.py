#coding:utf-8

class HyperParams(object):
    def __init__(self):
        pass


HP = HyperParams()


HP.embedding_size = 300
HP.hidden_size = 300
HP.n_layers = 3
HP.dropout_p = 0.2
HP.n_words = 30100
HP.max_word_len = 30
HP.tokens = dict({"PAD":0, "SOS":1, "EOS":2, "UNK":3})

HP.USE_CUDA = False
HP.learning_rate = 0.00005
HP.l2 = 1e-6
HP.batch_size = 32
HP.epoch = 40
HP.save_freq = 4000
HP.save_dir = "SavedModel/27"
HP.use_attention = True

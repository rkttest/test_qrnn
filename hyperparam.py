#coding:utf-8

class HyperParams(object):
    def __init__(self):
        pass


HP = HyperParams()


HP.embedding_size = 100 #400
HP.hidden_size = 100 #500
HP.n_layers = 2
HP.dropout_p = 0.2
HP.n_words = 7000 #55000
HP.max_word_len = 30
HP.tokens = dict({"PAD":0, "SOS":1, "EOS":2, "UNK":3})

HP.USE_CUDA = False
HP.learning_rate = 0.001
HP.l2 = 0.000001
HP.batch_size = 32
HP.epoch = 30
HP.save_freq = 4000
HP.save_dir = "SavedModel/16"
HP.use_attention = True

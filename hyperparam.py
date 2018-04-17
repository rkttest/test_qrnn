#coding:utf-8

class HyperParams(object):
    def __init__(self):
        pass


HP = HyperParams()


HP.embedding_size = 400
HP.hidden_size = 500
HP.n_layers = 2
HP.dropout_p = 0.2
HP.n_words = 55000
HP.max_word_len = 30
HP.tokens = dict({"PAD":0, "SOS":1, "EOS":2, "UNK":3})

HP.USE_CUDA = True
HP.learning_rate = 0.00025
HP.l2 = 0.000001
HP.batch_size = 28
HP.epoch = 30
HP.save_freq = 4000
HP.save_dir = "SavedModel/17"
HP.use_attention = True

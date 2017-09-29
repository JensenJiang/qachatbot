

class Config:
    def __init__(self, args_dict={}):
        # Basic parameters
        self.learning_rate = 0.1
        self.minibatch_size = 32
        self.max_epoch = 500
        self.epoch_instances = 9600
        self.keep_prob = 1.0

        # GAN parameters
        self.dis_learning_rate = 0.01
        self.gen_learning_rate = 0.01
        self.clip_min = -0.1
        self.clip_max = 0.1
        self.critic = 50

        # Dataset Location
        self.traindata_path = r'./data/bdwm_data_token.pkl'
        self.dict_path = r'./data/dict.txt'

        # RNN related parameters
        self.encoder_depth = 1
        self.decoder_depth = 1
        self.hidden_units = 512
        self.max_decode_step = 50

        # Embedding related parameters
        self.embedding_size = 512
        self.encoder_symbols_num = 20000
        self.decoder_symbols_num = 20000
        self._PAD = 0
        self._GO = 1
        self._EOS = 2

        # Attention related parameters
        self.attn_input_feeding = False

    @property
    def epoch_steps(self):
        return self.epoch_instances // self.minibatch_size

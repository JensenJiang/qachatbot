import os

import yaml

from madelight.utils import ModelLocator


class Config:
    def __init__(self, exp_name):
        # Basic parameters

        self.exp_name = exp_name
        self.minibatch_size = 128
        self.max_epoch = 100

        # Dataset Location
        self.traindata_path = r'./data/Short-Text-Conversation/q25a25_train.pkl'
        self.dict_path = r'./data/Short-Text-Conversation/model_256.vec'

        # RNN related parameters
        self.encoder_depth = 2
        self.decoder_depth = 2
        self.hidden_units = 512
        self.max_decode_step = 30

        # Embedding related parameters
        self.embedding_size = 256
        self.encoder_symbols_num = 20000
        self.decoder_symbols_num = 20000
        self._PAD = 0
        self._GO = 1
        self._EOS = 2

        # Attention related parameters
        self.attn_input_feeding = False

        # Model locator
        self.modloc = ModelLocator()

        # Load from external configuration file(default is yaml)
        self._exclude_from_save = ['modloc', '_exclude_from_save']
        self._load_from_ext()

    @property
    def epoch_steps(self):
        return self.epoch_instances // self.minibatch_size

    @property
    def _save_dict(self):
        return {k: v for k, v in vars(self).items() if k not in self._exclude_from_save}

    def print_config(self):
        print('---------------HYPERPARAMETERS---------------')
        for k, v in self._save_dict.items():
            print('{}: {}'.format(k, v))
        print('---------------------------------------------')

    def save_config_default_path(self):
        with open(os.path.join(self.modloc.exp_train_log_dir(self.exp_name), self.exp_name + '.yaml'), 'w') as fout:
            yaml.dump(self._save_dict, fout, default_flow_style=False)

    def _load_from_ext(self):
        with open(os.path.join('./experiment', self.exp_name + '.yaml'), 'r') as fin:
            ext_config = yaml.load(fin)
            if ext_config is not None:
                if 'exp_name' in ext_config:
                    assert ext_config['exp_name'] == self.exp_name, 'Experiment name is not consistent.'
                for k, v in ext_config.items():
                    setattr(self, k, v)

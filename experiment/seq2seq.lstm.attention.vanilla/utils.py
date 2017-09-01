import os
import pickle
from collections import defaultdict

import numpy as np

import tensorflow as tf
from tensorflow.python.summary.summary import FileWriter


class TrainClock:
    def __init__(self):
        self.global_step = 0
        self.cur_epoch = 0
        self.cur_epoch_step = 0

    def tick(self):
        self.global_step += 1
        self.cur_epoch_step += 1

    def tock(self):
        self.cur_epoch += 1
        self.cur_epoch_step = 0


class TrainHelper:
    def __init__(self, sess):
        # Training related paths
        self.train_log_path = os.path.join(os.path.curdir, r'train_log')
        self.models_path = os.path.join(self.train_log_path, r'models')

        # Training resources
        self.sess = sess
        self.saver = tf.train.Saver()
        self.clock = TrainClock()

        # Initialization
        self._init_log_dirs()

    def _init_log_dirs(self):
        # train_log path
        if not os.path.exists(self.train_log_path):
            os.makedirs(self.train_log_path)

        # models path
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    def save_checkpoint(self, name):
        ckpt_path = os.path.join(self.models_path, name)
        clock_path = os.path.join(self.models_path, name + r'.clock')
        self.saver.save(self.sess, ckpt_path)
        with open(clock_path, 'wb') as fout:
            pickle.dump(self.clock, fout)

    def load_checkpoint(self, ckpt_path):
        clock_path = ckpt_path + r'.clock'
        self.saver.restore(self.sess, ckpt_path)
        with open(clock_path, 'rb') as fin:
            self.clock = pickle.load(fin)


class GeneratorDataset:
    def __init__(self, gen):
        self._gen = gen

    def get_dataset_iter(self):
        yield from self._gen()


class InfiniteDataset:
    def __init__(self, dataset):
        self._dataset = dataset

    def get_dataset_iter(self):
        while True:
            yield from self._dataset.get_dataset_iter()


class EpochDataset:
    def __init__(self, dataset, instances_in_epoch):
        self._iter = InfiniteDataset(dataset).get_dataset_iter()
        self._instances_in_epoch = instances_in_epoch

    def get_dataset_iter(self):
        for i in range(self._instances_in_epoch):
            yield next(self._iter)


class MinibatchDataset:
    def __init__(self, dataset, minibatch_size, truncate=True):
        self._minibatch_size = minibatch_size
        self._dataset = dataset
        self._truncate = truncate

    def get_dataset_iter(self):
        list_dict = defaultdict(list)
        _count = 0

        for data in self._dataset.get_dataset_iter():
            _count += 1
            for k, v in data.items():
                list_dict[k].append(v)
            if _count == self._minibatch_size:
                kv = {}
                for k, vl in list_dict.items():
                    kv[k] = np.array(vl)
                yield kv
                list_dict.clear()
                _count = 0

        if _count > 0 and not self._truncate:
            kv = {}
            for k, vl in list_dict.items():
                kv[k] = np.array(vl)
            yield kv


class TensorBoardLogger:
    def __init__(self, path):
        self._writer = FileWriter(path, flush_secs=120)

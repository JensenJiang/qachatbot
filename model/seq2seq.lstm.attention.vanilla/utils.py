import contextlib
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

import tensorflow as tf
import tqdm
import yaml
from tensorflow.python.summary.summary import SessionLog, Summary
from tensorflow.python.summary.writer.writer import FileWriter


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


class ModelLocator:
    def __init__(self, model_dir=None):
        self.model_dir = os.path.curdir if model_dir is None else model_dir

    @property
    def root_train_log_dir(self):
        return os.path.join(self.model_dir, r'train_log')

    @property
    def exp_config_dir(self):
        return os.path.join(self.model_dir, r'experiment')

    def exp_train_log_dir(self, exp_name):
        return os.path.join(self.root_train_log_dir, exp_name)

    def ckpt_dir(self, exp_name):
        return os.path.join(self.exp_train_log_dir(exp_name), r'ckpt')


class TrainHelper:
    def __init__(self, sess, config):
        # Model locator
        self.modloc = config.modloc

        # Training related paths
        self.train_log_path = self.modloc.exp_train_log_dir(config.exp_name)
        self.ckpt_dir = self.modloc.ckpt_dir(config.exp_name)

        # Training resources
        self.sess = sess
        self.saver = tf.train.Saver(max_to_keep=None)  # Save all the checkpoints.
        self.clock = TrainClock()

        # Initialization
        self._init_log_dirs()

        # Dump config file and print
        config.save_config_default_path()
        config.print_config()

    def _init_log_dirs(self):
        # models path
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    def save_checkpoint(self, name):
        ckpt_path = os.path.join(self.ckpt_dir, name)
        clock_path = os.path.join(self.ckpt_dir, name + r'.clock')
        self.saver.save(self.sess, ckpt_path)
        with open(clock_path, 'wb') as fout:
            pickle.dump(self.clock, fout)

    def load_checkpoint(self, ckpt_path):
        clock_path = ckpt_path + r'.clock'
        self.saver.restore(self.sess, ckpt_path)
        with open(clock_path, 'rb') as fin:
            self.clock = pickle.load(fin)


class EvalHelper:
    def __init__(self, sess, config):
        self.sess = sess
        self.modloc = config.modloc
        self.saver = tf.train.Saver()
        self.ckpt_dir = self.modloc.ckpt_dir(config.exp_name)

    def load_checkpoint(self, name):
        self.saver.restore(self.sess, os.path.join(self.ckpt_dir, name))


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

    def put_start(self, global_step):
        self._writer.add_session_log(SessionLog(status=SessionLog.START), global_step)

    def put_scalar(self, k, v, global_step):
        self._writer.add_summary(Summary(value=[Summary.Value(tag=k, simple_value=float(v))]), global_step)


class DummyTqdmFile:
    def __init__(self, _file):
        self._file = _file

    def write(self, c):
        if len(c.rstrip()) > 0:
            tqdm.tqdm.write(c, file=self._file)


@contextlib.contextmanager
def stdout_redirect_to_tqdm():
    stdout_bak = sys.stdout
    try:
        sys.stdout = DummyTqdmFile(sys.stdout)
        yield stdout_bak
    except Exception as e:
        raise e
    finally:
        sys.stdout = stdout_bak

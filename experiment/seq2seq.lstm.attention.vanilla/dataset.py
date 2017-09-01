import pickle
from collections import defaultdict

import numpy as np

from utils import EpochDataset, GeneratorDataset, MinibatchDataset

_PAD = 0


class SeqPadMinibatchDataset(MinibatchDataset):
    def _build_qa(self, q_all, a_all, q_len, a_len, size):
        q_max_len = max(q_len)
        a_max_len = max(a_len)
        q = []
        a = []

        for i in range(size):
            q.append(np.pad(q_all[i],
                            (0, q_max_len - q_len[i]),
                            'constant',
                            constant_values=_PAD))
            a.append(np.pad(a_all[i],
                            (0, a_max_len - a_len[i]),
                            'constant',
                            constant_values=_PAD))

        return dict(
            encoder_inputs=np.array(q, dtype=np.int32),
            decoder_inputs=np.array(a, dtype=np.int32),
            encoder_inputs_length=np.array(q_len, dtype=np.int32),
            decoder_inputs_length=np.array(a_len, dtype=np.int32)
        )

    def get_dataset_iter(self):
        list_dict = defaultdict(list)
        _count = 0

        for data in self._dataset.get_dataset_iter():
            _count += 1
            for k, v in data.items():
                list_dict[k].append(v)
            if _count == self._minibatch_size:
                kv = self._build_qa(
                    list_dict['encoder_inputs'],
                    list_dict['decoder_inputs'],
                    list_dict['encoder_inputs_length'],
                    list_dict['decoder_inputs_length'],
                    _count
                )
                yield kv
                list_dict.clear()
                _count = 0

        if _count > 0 and not self._truncate:
            kv = self._build_qa(
                list_dict['encoder_inputs'],
                list_dict['decoder_inputs'],
                list_dict['encoder_inputs_length'],
                list_dict['decoder_inputs_length'],
                _count
            )
            yield kv


class DataProvider:
    def __init__(self, config):
        with open(config.traindata_path, 'rb') as fin:
            self.data = pickle.load(fin)

        self.config = config
        self.all_qa = [(tid, pid) for tid in self.data for pid in self.data[tid]]

    def _gen(self):
        np.random.shuffle(self.all_qa)

        for tid, pid in self.all_qa:
            q, a = self.data[tid][pid]
            yield dict(
                encoder_inputs=np.array(q, dtype=np.int32),
                decoder_inputs=np.array(a, dtype=np.int32),
                encoder_inputs_length=len(q),
                decoder_inputs_length=len(a)
            )

    def get(self):
        dataset = GeneratorDataset(self._gen)
        dataset = EpochDataset(dataset, self.config.epoch_instances)
        dataset = SeqPadMinibatchDataset(dataset, self.config.minibatch_size)
        return dataset

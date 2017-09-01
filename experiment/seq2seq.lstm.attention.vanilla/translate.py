import numpy as np

from tensorlayer.nlp import initialize_vocabulary


def seq2seq_onehot2label(data):
    data = np.array(data)
    seqlen, batch_size, _ = data.shape

    # (seqlen, batch_size, embedding_size) -> (batch_size, seqlen, embedding_size)
    data = np.transpose(data, (1, 0, 2))
    data = np.argmax(data, 2)
    data.reshape((batch_size, seqlen))
    return data


class Translator(object):
    def __init__(self, dict_path):
        _, self.my_dict = initialize_vocabulary(dict_path)

    def translate(self, data):
        r = []
        for sentence in data:
            r.append(list(map(lambda t: self.my_dict[t].decode('utf-8'), sentence)))

        return r

    def translate_and_print(self, data):
        for sentence in self.translate(data):
            print(' '.join(sentence))

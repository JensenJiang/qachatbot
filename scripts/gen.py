import argparse
import os
import pickle
from collections import defaultdict

from preprocess import bdwm_flaten, bdwm_qagen
from tensorlayer.nlp import (create_vocabulary, initialize_vocabulary,
                             sentence_to_token_ids)

if __name__ == '__main__':
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='data_name', help='data name, should be consistent with your split_pair file.')
    parser.add_argument(dest='max_vocab_size', type=int, help='maximum words in your dictionary.')
    parser.add_argument('-p', '--protocol', type=int, default=3, help='pickle protocol number.')
    args = parser.parse_args()

    # Data path
    data_name = args.data_name
    max_vocab_size = args.max_vocab_size
    data_dir = r'../data'
    ori_pkl = os.path.join(data_dir, '{}.split_pair.pkl'.format(data_name))
    flat_path = os.path.join(data_dir, '{}_{}.qaflat'.format(data_name, max_vocab_size))
    dict_path = os.path.join(data_dir, '{}_{}.dict'.format(data_name, max_vocab_size))
    token_pkl = os.path.join(data_dir, '{}_{}.token.pkl'.format(data_name, max_vocab_size))

    # Load data
    with open(ori_pkl, 'rb') as fin:
        data = pickle.load(fin)

    bdwm_flaten(data, flat_path)

    create_vocabulary(dict_path, flat_path, max_vocab_size)

    print('Next, tokenize the data')
    mydict, _ = initialize_vocabulary(dict_path)
    new_data = defaultdict(dict)

    for q, a, (tid, pid) in bdwm_qagen(data):
        new_data[tid][pid] = (
            sentence_to_token_ids(' '.join(q), mydict),
            sentence_to_token_ids(' '.join(a), mydict)
        )

    # Dump
    with open(token_pkl, 'wb') as fout:
        pickle.dump(new_data, fout, protocol=args.protocol)

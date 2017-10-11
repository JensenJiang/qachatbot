import argparse

import numpy as np
import tensorflow as tf

from config import Config
from dataset import DataProvider
from madelight.evaluate import EvalHelper
from madelight.utils.nlp import WordVecDict
from model import GANBasicModel


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='exp_name', help='Name of the experiment you want to evaluate.')
    parser.add_argument(dest='epoch_name', help='Epoch name')
    args = parser.parse_args()

    # Load Model
    config = Config(args.exp_name)
    with open(config.dict_path) as dict_in:
        wd = WordVecDict.from_vec_file(dict_in, exclude=['</s>'])

    # Load Data
    provider = DataProvider(config, 'test')
    dataset = provider.get()

    with tf.Session() as sess:
        model = GANBasicModel(config, wd.get_vecs_mat(), 'decode')
        helper = EvalHelper(sess, config)
        helper.load_checkpoint(args.epoch_name)
        fetch_dict = model.build_fetch_dict(['decoder_pred_decode'])
        for minibatch in dataset.get_dataset_iter():
            test_input = {i: minibatch[i] for i in ['encoder_inputs', 'encoder_inputs_length']}
            feed_dict = model.build_feed_dict(**test_input)
            outputs = sess.run(fetch_dict, feed_dict=feed_dict)
            words = wd.ids2words(outputs['decoder_pred_decode'])
            true_ans = wd.ids2words(minibatch['decoder_inputs'])
            qs = wd.ids2words(minibatch['encoder_inputs'])
            for batch in zip(qs, true_ans, words):
                for i in batch:
                    print(' '.join(i))
            input('Press Enter to Continue...')


if __name__ == '__main__':
    main()

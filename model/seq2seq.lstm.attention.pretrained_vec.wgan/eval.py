import argparse

import numpy as np
import tensorflow as tf

from config import Config
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

    with tf.Session() as sess:
        model = GANBasicModel(config, wd.get_vecs_mat(), 'decode')
        helper = EvalHelper(sess, config)
        helper.load_checkpoint(args.epoch_name)
        feed_dict = model.build_feed_dict(
            encoder_inputs=np.array([[1, 2, 3, 4]]),
            encoder_inputs_length=np.array([4])
        )
        fetch_dict = model.build_fetch_dict(['decoder_pred_decode'])
        outputs = sess.run(fetch_dict, feed_dict=feed_dict)
        print(outputs)


if __name__ == '__main__':
    main()

import argparse
import sys
import time

import tensorflow as tf
import tqdm

from config import Config
from dataset import DataProvider
from madelight.train import TrainHelper
from madelight.utils.itqdm import stdout_redirect_to_tqdm
from madelight.utils.nlp import WordVecDict
from model import Seq2SeqBasicModel


def main():
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='exp_name', type=str, help='Name of the current experiment.')
    parser.add_argument('-c', '--continue', dest='continue_path', type=str, help='Continue a given checkpoint.')
    args = parser.parse_args()

    config = Config(args.exp_name)
    provider = DataProvider(config, 'train')
    with open(config.dict_path) as dict_in:
        wd = WordVecDict.from_vec_file(dict_in, exclude=['</s>'])
    model = Seq2SeqBasicModel(config, wd.get_vecs_mat(), 'train')

    with tf.Session() as sess:
        helper = TrainHelper(sess, config)
        sess.run(tf.global_variables_initializer())

        # Restore checkpoint
        if args.continue_path is not None:
            helper.load_checkpoint(args.continue_path)

        clock = helper.clock

        # Print model size
        tot_p_num = 0
        for p in tf.trainable_variables():
            cur = 1
            for i in p.get_shape():
                cur *= i.value
            tot_p_num += cur
        print(tot_p_num)

        # Get dataset
        dataset = provider.get()

        # Tensorboard
        train_tb, = helper.create_tbs('train.tb')
        train_tb.put_start(clock.global_step)

        while clock.cur_epoch < config.max_epoch:
            print('Starting epoch {}:'.format(clock.cur_epoch))

            # Adjust learning rate
            if clock.cur_epoch < 6:
                lr = 1e-3
            elif clock.cur_epoch < 10:
                lr = 1e-4
            elif clock.cur_epoch < 18:
                lr = 1e-5
            else:
                lr = 1e-6

            # Main loop
            data_wrapper = tqdm.tqdm(dataset.get_dataset_iter(), total=config.epoch_steps, file=sys.stdout)
            data_wrapper.set_description('Epoch {}'.format(clock.cur_epoch))
            data_wrapper.set_postfix(learning_rate=lr)
            for minibatch in data_wrapper:
                with stdout_redirect_to_tqdm():
                    feed_dict = model.build_feed_dict(**minibatch)
                    feed_dict[model.learning_rate] = lr
                    fetch_dict = model.build_fetch_dict(['loss', 'updates'])
                    outputs = sess.run(fetch_dict, feed_dict=feed_dict)
                    print('loss: {}'.format(outputs['loss']))

                train_tb.put_scalar('seq_loss', outputs['loss'], clock.global_step)
                clock.tick()

                if clock.global_step % 200 == 0:
                    train_tb.flush()

            clock.tock()

            # Save checkpoint
            if clock.cur_epoch % 2 == 0:
                helper.save_checkpoint('epoch_{}'.format(clock.cur_epoch))
            helper.save_checkpoint('latest')


if __name__ == '__main__':
    main()

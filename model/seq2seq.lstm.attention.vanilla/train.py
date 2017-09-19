import argparse
import sys
import time

import tensorflow as tf
import tqdm
from config import Config
from dataset import DataProvider
from model import Seq2SeqBasicModel
from utils import TrainHelper, stdout_redirect_to_tqdm


def main():
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='exp_name', type=str, help='Name of the current experiment.')
    parser.add_argument('-c', '--continue', dest='continue_path', type=str, help='Continue a given checkpoint.')
    args = parser.parse_args()

    config = Config(args.exp_name)
    provider = DataProvider(config)
    model = Seq2SeqBasicModel(config, 'train')

    with tf.Session() as sess:
        helper = TrainHelper(sess, config)
        clock = helper.clock
        sess.run(tf.global_variables_initializer())

        # Restore checkpoint
        if args.continue_path is not None:
            helper.load_checkpoint(args.continue_path)

        start_time = time.time()

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

        while clock.cur_epoch < config.max_epoch:
            print('Starting epoch {}:'.format(clock.cur_epoch))

            # Adjust learning rate
            pass

            # Main loop
            for minibatch in tqdm.tqdm(dataset.get_dataset_iter(), total=config.epoch_steps, file=sys.stdout):
                with stdout_redirect_to_tqdm():
                    feed_dict = model.build_feed_dict(**minibatch)
                    fetch_dict = model.build_fetch_dict(['loss', 'updates'])
                    outputs = sess.run(fetch_dict, feed_dict=feed_dict)
                    print('loss: {}'.format(outputs['loss']))

                clock.tick()

            clock.tock()

            # Save checkpoint
            if clock.cur_epoch % 5 == 0:
                helper.save_checkpoint('epoch_{}'.format(clock.cur_epoch))
            helper.save_checkpoint('latest')


if __name__ == '__main__':
    main()

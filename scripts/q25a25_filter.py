import os
import pickle
from collections import defaultdict

from preprocess import bdwm_qagen

if __name__ == '__main__':
    # Data path
    data_dir = r'../data'
    ori_pkl = os.path.join(data_dir, 'all.split_pair.pkl')
    rst_pkl = os.path.join(data_dir, 'q25a25.split_pair.pkl')

    with open(ori_pkl, 'rb') as fin:
        data = pickle.load(fin)
    new_data = defaultdict(dict)

    count = 0
    for q, a, (tid, pid) in bdwm_qagen(data):
        if len(q) <= 25 and len(a) <= 25:
            count += 1
            new_data[tid][pid] = (q, a)

    print('Total: {} qas'.format(count))
    with open(rst_pkl, 'wb') as fout:
        pickle.dump(new_data, fout)

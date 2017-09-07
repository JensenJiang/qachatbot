

def bdwm_qagen(data):
    for tid in data:
        for pid in data[tid]:
            yield data[tid][pid][0], data[tid][pid][1], (tid, pid)


def bdwm_flaten(data, txt_path):
    with open(txt_path, 'w') as fout:
        for q, a, _ in bdwm_qagen(data):
            fout.write(' '.join(q).rstrip() + '\n')
            fout.write(' '.join(a).rstrip() + '\n')

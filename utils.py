import inspect, shelve, gp, cythondir.vm
import numpy as np
from importlib import reload


def conv_data_file(filename, newfile):
    with open(filename, 'r+') as f:
        with open(newfile, 'w+') as nf:
            for line in f.readlines():
                l = line.rstrip()
                l = l.replace(' ', ',')
                nf.write('{}\n'.format(l))


def filenum():
    with shelve.open('filegen') as c:
        try:
            c['num'] += 1
        except KeyError:
            c['num'] = 0
        n = c['num']
    return n


def reset_filenum():
    with shelve.open('filegen') as c:
        c['num'] = 0


def set_arr(index, arr, vals):
    for i in range(len(index)):
        try:
            arr[index[i]] = vals[i]
        except (TypeError, IndexError):
            arr[index[i]] = vals


def get_nonzero(arr):
    return arr[np.nonzero(arr)[0]]


def get_ranked_index(results):
    return [x[0] for x in sorted(enumerate(results), key=lambda i: i[1])]
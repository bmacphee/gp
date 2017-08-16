import shelve, pdb, random
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
            val, ind = vals[i], index[i]
            # if isinstance(val, Host):
            #     val.index_num = ind
            arr[ind] = val
        except (TypeError, IndexError):
            arr[index[i]] = vals


def get_nonzero(arr):
    return arr[np.nonzero(arr)[0]]


def get_ranked_index(results):
    return [x[0] for x in sorted(enumerate(results), key=lambda i: i[1])]


def top_host_i(stats, system):
    top = get_ranked_index(stats.trainset_with_testfit)[-1]
    ind = system.root_hosts()[top]
    return ind


def get_non_root(system):
    curr_inds = [np.where(host == system.hosts)[0][0] for host in system.curr_hosts()]
    return [i for i in curr_inds if i not in system.root_hosts()]


def get_non_atomic(system):
    return [i for i in range(len(system.pop)) if system.pop[i] is not None and system.pop[i].atomic_action == 0]


def prob_check(prob):
    return random.random() <= prob

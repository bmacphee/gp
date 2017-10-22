import shelve, pdb, random, time,os
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
        arr[index[i]] = vals[i]


def get_nonzero(arr):
    return arr[np.nonzero(arr)[0]]


def get_ranked_index(results, root_hosts_i=None):
    index =  [x[0] for x in sorted(enumerate(results), key=lambda i: i[1])]
    if root_hosts_i is not None:
        index = [root_hosts_i[i] for i in index]
    return index


def top_host_i(stats, system):
    top = get_ranked_index(stats.trainset_with_testfit, root_hosts_i=system.root_hosts)[-1]
    return top


def get_non_root(system):
    curr_inds = [np.where(host == system.hosts)[0][0] for host in system.curr_hosts]
    return [i for i in curr_inds if i not in system.root_hosts]


def get_non_atomic(system):
    return [i for i in range(len(system.pop)) if system.pop[i] is not None and system.pop[i].atomic_action == 0]


def prob_check(prob):
    return random.random() <= prob

def should_save(i, save_step, generations):
    return (save_step is not None and ((i % save_step == 0) or (i == generations - 1)))

def make_filename(dir, prefix, name):
    date = time.strftime("%d_%m_%Y")
    filepath = os.path.join(dir, date, '{}_{}'.format(prefix, name))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    return filepath
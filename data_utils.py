import const, matplotlib, gzip, time, os, pdb, systems, utils, pickle
import numpy as np, jsonpickle as jp
import jsonpickle.ext.numpy as jsonpickle_numpy

matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from cythondir.vm import Prog, Host
from array import array
from copy import deepcopy
from cythondir.vm import init
import fitness as fit
from systems import System

jsonpickle_numpy.register_handlers()


class Results:
    def __init__(self):
        self.percentages = {}
        self.trainset_with_trainfit = []
        self.trainset_with_testfit = []
        self.testset_with_testfit = []
        self.trainfit_trainset_means = []
        self.testfit_trainset_means = []
        self.top_i = None
        self.top_trainfit_in_trainset = []
        self.top_testfit_in_trainset = []
        self.last_y_pred = []
        self.num_progs_per_top_host = []
        self.num_hosts_in_graph = []
        self.active_progs_per_top_host = []
        self.num_effective_instrs = []
        self.max_fitness = 0
        self.max_fitness_gen = 0
        self.generation = 0

    def init_percentages(self, classes):
        for cl in classes:
            self.percentages[cl] = []

    def update_trainfit_trainset(self, trainset_with_trainfit):
        self.top_trainfit_in_trainset.append(max(trainset_with_trainfit))
        self.trainfit_trainset_means.append(np.mean(trainset_with_trainfit))

    def update_testfit_trainset(self, trainset_with_testfit, system):
        self.trainset_with_testfit = trainset_with_testfit
        self.top_testfit_in_trainset.append(max(trainset_with_testfit))
        self.testfit_trainset_means.append(np.mean(trainset_with_testfit))
        top_result_i = utils.get_ranked_index(trainset_with_testfit)[-1]
        self.top_i = system.root_hosts[top_result_i]

    def update_testfit_testset(self, testset_with_testfit, i):
        self.testset_with_testfit.append(testset_with_testfit)
        if testset_with_testfit > self.max_fitness:
            self.max_fitness = testset_with_testfit
            self.max_fitness_gen = i

    def update_percentages(self, class_percs):
        for cl in class_percs:
            self.percentages[cl].append(class_percs[cl])

    def update_prog_num(self, top_host, nodes=None):
        if not nodes:
            self.num_progs_per_top_host.append(top_host.progs_i.size)
            self.active_progs_per_top_host.append(len(top_host.get_active()))
            return

        # Graphs - do average (currently uses node format for printing graphs)
        hosts = [x for x in nodes if list(x[0])[0] == 'h']
        progs = [x for x in nodes if list(x[0])[0] == 's']
        self.num_progs_per_top_host.append(len(progs) / len(hosts))
        self.num_hosts_in_graph.append(len(hosts))

    def update_eff_instrs(self, progs):
        pass

    def get_graph_params(self, to_graph):
        graph_param = [self.top_trainfit_in_trainset, self.trainfit_trainset_means, self.testfit_trainset_means,
                       self.testset_with_testfit, self.top_testfit_in_trainset]
        graph_inc = [to_graph['top_trainfit_in_trainset'], to_graph['train_means'], to_graph['test_means'],
                     to_graph['top_train_prog_on_test'], to_graph['top_test_fit_on_train']]
        graph_param = list(map(lambda x: graph_param[x] if graph_inc[x] else None, range(len(graph_param))))
        return graph_param

    def save_objs(self, system, data, env):
        filepath = utils.make_filename(const.JSON_DIR, env.file_prefix, 'saved')
        save_system = SaveableSystem(system)
        save_data = deepcopy(data)
        save_env = deepcopy(env)
        set_none = ['X_train', 'y_train', 'X_test', 'y_test']
        for attr in set_none:
            setattr(save_data, attr, None)
        # Need last_X_train, or curr_ypred_state?
        convert_to_list = ['curr_X', 'data_i', 'curr_y', 'last_X_train', 'validation_X', 'validation_y']
        for attr in convert_to_list:
            val = getattr(save_data, attr)
            if val is not None:
                setattr(save_data, attr, val.tolist())
        save_env.ops = save_env.ops.tolist()

        objs = jp.encode([save_system, save_data, save_env, self])
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as outfile:
            outfile.write(objs)


class Data:
    def __init__(self):
        self.classes = None  # Available classes
        self.data_by_classes = {}  # X_train indices organized by classes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.last_X_train = None
        self.curr_X = None
        self.curr_y = None
        self.data_i = None
        self.act_subset_size = None
        self.act_valid_size = None
        self.validation_X = None
        self.validation_y = None
        self.validation_i = None

        self.max_vals = None
        self.grid = None
        self.num_ipregs = None
        self.output_dims = None

    def load_data(self, config, from_saved=False):
        if config.data_file is None:
            raise AttributeError('No data file to load')

        # MNIST data
        if config.data_file == config.data_files[4]:
            self.X_train, self.y_train, self.X_test, self.y_test = load_mnist()  # Requires gzip method
            self.classes = get_classes(self.y_train)

        # SVHN data
        elif config.data_file == config.data_files[6]:
            import scipy.io as sio
            from PIL import Image
            train = sio.loadmat(config.data_file)
            test = sio.loadmat(const.TEST_DATA_FILES[config.data_file])
            self.X_train = np.array([x.flatten() for x in train['X']], dtype=np.double)
            self.X_test = np.array([x.flatten() for x in test['X']], dtype=np.double)
            self.classes = get_classes(train['y'])
            self.y_train = convert_labels(train['y'].flatten(), self.classes)
            self.y_test = convert_labels(test['y'].flatten(), self.classes)

        # Temporary
        elif config.data_file == config.data_files[5]:
            import mnist
            # classes = [0, 1, 2, 3, 4, 5, 6, 7, 8,9]
            classes = [0, 1, 7, 8, 9]
            self.y_train = np.array(mnist.train_labels(), dtype=np.int32)
            self.y_test = np.array(mnist.test_labels(), dtype=np.int32)
            self.X_train = np.array([x.flatten() for x in mnist.train_images()], dtype=np.double)
            self.X_test = np.array([x.flatten() for x in mnist.test_images()], dtype=np.double)

            train_inds = [i for i in range(len(self.y_train)) if self.y_train[i] in classes]
            test_inds = [i for i in range(len(self.y_test)) if self.y_test[i] in classes]
            self.y_train = self.y_train[train_inds]
            self.y_test = self.y_test[test_inds]
            self.X_train = self.X_train[train_inds]
            self.X_test = self.X_test[test_inds]
            self.classes = get_classes(self.y_train)
            self.y_train = convert_labels(self.y_train, self.classes)
            self.y_test = convert_labels(self.y_test, self.classes)

        else:
            data = load_data(config.data_file)
            labels = data[:, -1]
            self.classes = get_classes(labels)
            X_train, y_train = split_data_labels(data, self.classes)

            # Data with corresponding test file - load train/test
            try:
                test_data_file = const.TEST_DATA_FILES[config.data_file]
                test_data = load_data(test_data_file)
                X_test, y_test = split_data_labels(test_data, self.classes)

            # Data with no corresponding test file - split into train/test
            except KeyError:
                X_train, X_test, y_train, y_test = split_data(X_train, y_train, config.test_size)

            self.X_train, self.X_test = standardize(X_train, X_test, config.standardize_method)
            self.y_train, self.y_test = y_train, y_test


        if config.use_validation:
            if from_saved:
                # Validation data should be already loaded
                if not self.validation_i:
                    temp_X = self.X_train.tolist()
                    temp_X_valid = self.validation_X.tolist()
                    valid_inds = []
                    for ex in temp_X_valid:
                        valid_inds.append(temp_X.index(ex))
                    inds = [i for i in range(len(self.X_train)) if i not in valid_inds]
                    self.X_train = self.X_train[inds]
                    self.y_train = self.y_train[inds]

                else:
                    inds = [i for i in range(len(self.X_train)) if i not in self.validation_i]
                    self.X_train = self.X_train[inds]
                    self.y_train = self.y_train[inds]

                pdb.set_trace()

            else:
                if len(self.X_train) < config.validation_size:
                    print("Turning off validation")
                else:
                    valid_X, valid_y, valid_i = even_data_subset(self, config.validation_size)
                    inds = [a for a in range(len(self.X_train)) if a not in valid_i]

                    self.X_train = self.X_train[inds]
                    # Change data type?
                    self.y_train = array('i', [self.y_train[i] for i in inds])
                    self.validation_X = valid_X
                    self.validation_y = valid_y
                    self.validation_i = valid_i
                    self.set_classes(self.X_train, self.y_train)
                    self.act_valid_size = len(valid_X)

        self.num_ipregs = len(self.X_train[0])
        self.max_vals = array('i', [const.GEN_REGS, max(const.GEN_REGS, self.num_ipregs), -1, 2])
        self.output_dims = 1 if config.bid_gp else len(self.classes)

        if config.use_subset and (len(self.X_train) < config.subset_size):
            print('Length of X_Train ({}) smaller than subset size - turning off subset'.format(len(self.X_train)))
            config.use_subset = 0
            config.use_validation = 0

    def set_classes(self, X, y):
        for cl in self.classes.values():
            self.data_by_classes[cl] = [i for i in range(len(X)) if y[i] == cl]


def convert_labels(y, classes):
    y_conv = np.array([classes[i] for i in y], dtype=np.int32)
    return y_conv


class SaveableProg:
    def __init__(self, prog):
        self.is_none = 0
        if prog is None:
            self.is_none = 1
        else:
            self.prog = prog.prog.base.tolist()
            self.atomic_action = prog.atomic_action
            self.class_label = prog.class_label
            self.prog_id = prog.prog_id
            self.grid_section = prog.grid_section

    def convert_to_prog(self):
        if self.is_none:
            return None

        new_prog = Prog(self.prog)
        new_prog.action = [self.atomic_action, self.class_label]
        new_prog.prog_id = self.prog_id
        new_prog.grid_section = self.grid_section
        return new_prog


class SaveableHost:
    def __init__(self, host):
        self.is_none = 0
        if host is None:
            self.is_none = 1
        else:
            self.progs_i = host.progs_i.base.tolist()
            self.index_num = host.index_num
            self.atomic_actions_allowed = set(host.atomic_actions_allowed)

    def convert_to_host(self):
        if self.is_none:
            return None

        new_host = Host()
        new_host.set_progs(array('i', self.progs_i))
        new_host.index_num = self.index_num
        new_host.atomic_actions_allowed = set(self.atomic_actions_allowed)
        return new_host


class SaveableSystem:
    def __init__(self, system):
        self.hosts = [SaveableHost(s) for s in system.hosts] if system.hosts is not None else None
        self.pop = [SaveableProg(p) for p in system.pop]
        self.cl = system.__class__

    def convert_to_system(self):
        for i, host in enumerate(self.hosts):
            self.hosts[i] = host.convert_to_host()
        System.hosts = self.hosts
        for i, p in enumerate(self.pop):
            self.pop[i] = p.convert_to_prog()
        system = self.cl(np.asarray(self.pop), np.asarray(self.hosts))
        return system


def get_host_class_percs(system, data, results):
    top = utils.get_ranked_index(results.trainset_with_testfit)
    percs = []
    for ind in reversed(top):
        percs.append(fit.class_percentages(system, data.X_test, data.y_test, data.classes, 1,
                                           array('i', [system.root_hosts()[ind]]),
                                           array('i', range(len(data.X_test)))))
    for perc in percs:
        print(perc)
    return percs


def load_saved(filename):
    with open(filename, 'r') as f:
        objs = f.read()
    decoded = jp.decode(objs)
    data = decoded[1]
    env = decoded[2]
    results = decoded[3]
    env.ops = array('i', env.ops)
    env.data_files = {int(k): v for k, v in env.data_files.items()}
    data.data_by_classes = {int(k): v for k, v in data.data_by_classes.items()}
    data.curr_X = np.asarray(data.curr_X)
    data.curr_y = array('i', data.curr_y)
    data.curr_y = array('i', data.curr_y)
    data.data_i = array('i', data.data_i)
    data.last_X_train = array('i', data.last_X_train)
    if env.use_validation:
        data.validation_X = np.asarray(data.validation_X)
        data.validation_y = array('i', data.data_i)
    data.load_data(env, from_saved=True)
    # Important - before converting programs
    init(const.GEN_REGS, data.output_dims, env.bid_gp, env.tangled_graphs, len(data.X_train), len(data.X_test))
    system = decoded[0].convert_to_system()
    return system, data, env, results


# Initializing data
def load_data(fname, split=','):
    return np.loadtxt(fname, delimiter=split, dtype=object)


def load_mnist():
    classes = set(range(10))
    #classes = set([0,1,2,8,9])
    X_train = np.frombuffer(gzip.open(const.MNIST_DATA_FILES['X_train'], 'rb').read(), dtype=np.float64).reshape(60000,
                                                                                                                 784)
    X_test = np.frombuffer(gzip.open(const.MNIST_DATA_FILES['X_test'], 'rb').read(), dtype=np.float64).reshape(10000,
                                                                                                               784)
    y_train = np.frombuffer(gzip.open(const.MNIST_DATA_FILES['y_train'], 'rb').read(), dtype=np.uint8).astype(np.intc)
    y_test = np.frombuffer(gzip.open(const.MNIST_DATA_FILES['y_test'], 'rb').read(), dtype=np.uint8).astype(np.intc)

    if classes != set(range(10)):
        train_inds = [i for i in range(len(y_train)) if y_train[i] in classes]
        test_inds = [i for i in range(len(y_test)) if y_test[i] in classes]
        X_train = X_train[train_inds]
        X_test = X_test[test_inds]
        y_train = y_train[train_inds]
        y_test = y_test[test_inds]

    return_vals = [X_train, y_train, X_test, y_test]
    for arr in return_vals:
        arr.flags.writeable = True
    # Could use a generator?
    # return_vals = [(x for x in X_train), (y for y in y_train), (x for x in X_test), (y for y in y_test)]


    return return_vals


def get_classes(labels):
    classes = np.unique(labels)
    class_dict = {}
    for i in range(len(classes)):
        if (type(classes[i]) == int) or (type(classes[i]) == np.int32):
            class_dict[classes[i]] = classes[i]
        else:
            class_dict[classes[i]] = i
    #return {classes[i]: i for i in range(len(classes))}
    return class_dict

def standardize(train, test, method, alpha=1):
    if method is None:
        return [train, test]

    num_attrs = len(train[0])
    vals0, vals1 = [None] * num_attrs, [None] * num_attrs
    standardized = []
    matrices = [train, test]
    for m in matrices:
        m_transpose = np.transpose(m)
        if method is const.StandardizeMethod.MEAN_VARIANCE:
            # vals0 = std, vals1 = mean
            for col in range(num_attrs):
                if vals0[col] is None:
                    vals0[col], vals1[col] = np.std(m_transpose[col]), np.mean(m_transpose[col])
                std, mean = vals0[col], vals1[col]
                for row in range(len(m)):
                    if std != 0:
                        m[row, col] = alpha * ((m.item(row, col) - mean) / std)
                    else:
                        m[row, col] = 0
            standardized.append(m)
        elif method is const.StandardizeMethod.LINEAR_TRANSFORM:
            # vals0 = min_x, vals1 = max_x
            for col in range(num_attrs):
                if vals0[col] is None:
                    vals0[col], vals1[col] = min(m_transpose[col].tolist()[0]), max(m_transpose[col].tolist()[0])
                min_x, max_x = vals0[col], vals1[col]
                for row in range(len(m)):
                    denom = max_x - min_x
                    if denom != 0:
                        m[row, col] = alpha * ((m.item(row, col) - min_x) / denom)
                    else:
                        m[row, col] = 0
            standardized.append(m)
        elif method is const.StandardizeMethod.MIN_MAX_VAL:
            pass
        else:
            raise AttributeError('Invalid standardize method')
    return standardized


def preprocess(data):
    while True:
        try:
            return data.astype(np.double)
        except ValueError:
            attrs = np.unique(data).tolist()
            for attr in attrs:
                data[np.where(data == attr)] = attrs.index(attr)


def split_data_labels(data, classes):
    y = np.array([classes[label] for label in data[:, -1]], dtype=np.int32)
    data = np.delete(data, -1, 1)
    X = preprocess(data)
    return (X, y)


# import scipy.io as sio
# from PIL import Image
# from skimage import color
# from skimage import exposure
# f0="C:\\Users\\S\\Dropbox\\1\\data\\SVHN_test_32x32.mat"
# f1="C:\\Users\\S\\Dropbox\\1\\data\\SVHN_train_32x32.mat"
# d=sio.loadmat(f0)
# x, y = d['X'],d['y']
# pics = [x[:,:,:,i] for i in range(x.shape[-1])]
# picsg = [color.rgb2gray(X) for X in pics]
# picsge = [exposure.equalize_hist(p) for p in picsg]
#
# def get_vals(mdict):
#     X,y = mdict['X'], mdict['y']
#     picsg = [color.rgb2gray(x) for x in X]
#     new_d={}
#     new_d['X'] = np.array(picsg)
#     new_d['y'] = y
#     return new_d

def save_gscale(X, filename):
    from PIL import Image
    picsg = np.array([color.rgb2gray(x) for x in X])
    np.save(filename, picsg)
    return picsg


def convert_svhn(X_train, X_test, y_train, y_test):
    num_test = X_test.shape[-1]
    num_train = X_train.shape[-1]
    train = [X_train[:, :, :, i] for i in range(num_train)]
    test = [X_test[:, :, :, i] for i in range(num_test)]


def save_preprocessed(X_train, X_test, y_train, y_test, normalize_func):
    for y in [y_train, y_test]:
        while y.ndim > 1:
            y = y.flatten()

    for data in [X_train, X_test]:
        for x in data:
            while x.ndim > 1:
                x = x.flatten()


def split_data(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    return X_train, X_test, array('i', y_train), array('i', y_test)


def even_data_subset(data, subset_size):
    if not data.data_by_classes:
        data.set_classes(data.X_train, data.y_train)
    data_by_classes = data.data_by_classes
    orig_subset_size = int(subset_size / len(data_by_classes))
    assert orig_subset_size > 0, 'Subset size not large enough for number of classes: {}'.format(len(data_by_classes))
    subset_x, subset_y, x_ind = [], [], []
    for i in data_by_classes:
        subset_size = orig_subset_size
        class_size = len(data_by_classes[i])
        if class_size <= subset_size:
            subset_size = class_size
            x_ind += data_by_classes[i]
        else:
            x_ind += np.random.choice(data_by_classes[i], size=subset_size, replace=False).tolist()
        subset_y += [i] * subset_size
    subset_x = [data.X_train[i] for i in x_ind]
    return np.array(subset_x), array('i', subset_y), array('i', x_ind)


def uniformprob_data_subset(data, subset_size):
    X = data.X_train
    x_ind = array('i', np.random.choice(range(len(X)), subset_size))
    subset_x = np.array([data.X_train[i] for i in x_ind])
    subset_y = array('i', [data.y_train[i] for i in x_ind])
    if not data.act_subset_size:
        data.act_subset_size = len(subset_x)
    return subset_x, subset_y, x_ind


def get_top_host_graph(system, env, data, stats):
    assert type(system) is systems.GraphSystem, 'GraphSystem required'
    top = utils.top_host_i(stats, system)
    return get_nodes_host(system, top, data, env.file_prefix)


def get_nodes_host(system, host_i, data, filenum, path_i=None):
    nodes = set()
    edges = set()
    path = set()
    node = host_n(host_i)
    if path_i is not None:
        path.add(node)
    nodes.add(node)
    for i, ex in enumerate(data.X_test):
        get_traversed(system, host_i, ex, nodes, edges, node, i is path_i, path)
    return nodes, edges, path


def get_traversed(system, host_i, X, nodes, edges, last_node, on_path, path=None):
    vals = []
    for prog_i in system.hosts[host_i].progs_i:
        prog = system.pop[prog_i]
        prog.run_prog(X)
        vals.append(prog.get_regs()[0])
    winner_i = system.hosts[host_i].progs_i[vals.index(max(vals))]
    winner = system.pop[winner_i]
    node0 = prog_n(winner.prog_id)
    nodes.add(node0)
    edges.add((last_node[0], node0[0]))
    if on_path:
        path.add(node0)
    if winner.atomic_action == 1:
        node1 = action_n(winner.prog_id, winner.class_label)
    else:
        node1 = host_n(winner.class_label)
        if on_path:
            path.add(node1)
    nodes.add(node1)
    edges.add((node0[0], node1[0]))
    if winner.atomic_action == 0:
        get_traversed(system, winner.class_label, X, nodes, edges, node1, on_path, path)


def host_n(id):
    return ('h{}'.format(id), 'host')


def prog_n(id):
    return ('s{}'.format(id), 'symbiont')


def action_n(id, class_label):
    return ('a{}'.format(id), 'atomic-{}'.format(class_label))

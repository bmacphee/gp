import const, matplotlib, gzip, time, os, pdb
import numpy as np, jsonpickle as jp

matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from cythondir.vm import Prog, Host
from array import array
from copy import deepcopy


class Results:
    def __init__(self):
        self.percentages = {}
        self.trainset_with_trainfit = []
        self.trainset_with_testfit = []
        self.testset_with_testfit = []
        self.trainfit_trainset_means = []
        self.testfit_trainset_means = []
        self.top_trainfit_in_trainset = []
        self.top_testfit_in_trainset = []
        self.num_progs_per_top_host = []
        self.num_effective_instrs = []
        self.max_fitness = 0
        self.max_fitness_gen = 0

    def init_percentages(self, classes):
        for cl in classes:
            self.percentages[cl] = []

    def update_trainfit_trainset(self, trainset_with_trainfit):
        self.top_trainfit_in_trainset.append(max(trainset_with_trainfit))
        self.trainfit_trainset_means.append(np.mean(trainset_with_trainfit))

    def update_testfit_trainset(self, trainset_with_testfit):
        self.trainset_with_testfit = trainset_with_testfit
        self.top_testfit_in_trainset.append(max(trainset_with_testfit))
        self.testfit_trainset_means.append(np.mean(trainset_with_testfit))

    def update_testfit_testset(self, testset_with_testfit, i):
        self.testset_with_testfit.append(testset_with_testfit)
        if testset_with_testfit > self.max_fitness:
            self.max_fitness, self.max_fitness_gen = testset_with_testfit, i

    def update_percentages(self, class_percs):
        for cl in class_percs:
            self.percentages[cl].append(class_percs[cl])

    def update_prog_num(self, top_host):
        if top_host is not None:
            self.num_progs_per_top_host.append(top_host.progs_i.size)

    def update_eff_instrs(self, progs):
        pass

    def get_graph_params(self, to_graph):
        graph_param = [self.top_trainfit_in_trainset, self.trainfit_trainset_means, self.testfit_trainset_means,
                       self.testset_with_testfit, self.top_testfit_in_trainset]
        graph_inc = [to_graph['top_trainfit_in_trainset'], to_graph['train_means'], to_graph['test_means'],
                     to_graph['top_train_prog_on_test'], to_graph['top_test_fit_on_train']]
        graph_param = list(map(lambda x: graph_param[x] if graph_inc[x] else None, range(len(graph_param))))
        return graph_param

    def save_objs(self, pop, hosts, data, env):
        file_name = '{}_saved_data'.format(env.file_prefix)
        date = time.strftime("%d_%m_%Y")
        filepath = os.path.join(const.JSON_DIR, date, file_name)
        save_pop = [SaveableProg(p) for p in pop]
        save_hosts = [SaveableHost(h) for h in hosts]
        save_data = deepcopy(data)
        save_data.X_train = None
        save_data.y_train = None
        save_data.X_test = None
        save_data.y_test = None
        # Need last_X_train, or curr_ypred_state?
        save_data.curr_X = save_data.curr_X.tolist()
        save_data.curr_i = save_data.curr_i.tolist()
        save_data.curr_y = save_data.curr_y.tolist()
        save_env = deepcopy(env)
        save_env.ops = save_env.ops.tolist()

        objs = jp.encode([save_pop, save_hosts, save_data, save_env, self])

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
        self.curr_i = None
        self.act_subset_size = None
        self.act_valid_size = None

    def load_data(self, config):
        if config.data_file is None:
            raise AttributeError('No data file to load')

        if config.data_file == 'data/MNIST':
            self.X_train, self.y_train, self.X_test, self.y_test = load_mnist()  # Requires gzip method
            self.classes = get_classes(self.y_train)

        else:
            data = load_data(config.data_file)
            try:
                labels = load_data(const.LABEL_FILES[config.data_file])
                y = [int(i) for sublist in labels for i in sublist]
                X = preprocess([ex for ex in data])
            except KeyError:
                y = [ex[-1:len(ex)][0] for ex in data]
                X = preprocess([ex[:len(ex) - 1] for ex in data])
            self.classes = get_classes(y)
            y = np.array([self.classes[label] for label in y], dtype=np.int32)

            # Data with corresponding test file - load train/test
            try:
                test_data_file = const.TEST_DATA_FILES[config.data_file]
                test_data = load_data(test_data_file)
                X_train = X
                self.y_train = y

                try:
                    X_test = preprocess([ex for ex in test_data])
                    labels = load_data(const.LABEL_FILES[test_data_file])
                    self.y_test = [self.classes[int(i)] for sublist in labels for i in sublist]
                except KeyError:
                    X_test = preprocess([ex[:len(ex) - 1] for ex in test_data])
                    self.y_test = np.array([self.classes[label] for label in [ex[-1:len(ex)][0] for ex in test_data]],
                                           dtype=np.int32)
            # Data with no corresponding test file - split into train/test
            except KeyError:
                X_train, X_test, self.y_train, self.y_test = split_data(X, y, config.test_size)

            if config.standardize_method is not None:
                X_train, X_test = standardize(X_train, X_test, config.standardize_method)
            self.X_train, self.X_test = np.array(X_train, dtype=np.float64), np.array(X_test, dtype=np.float64)

        config.num_ipregs = len(self.X_train[0])
        config.max_vals = array('i', [const.GEN_REGS, max(const.GEN_REGS, config.num_ipregs), -1, 2])
        config.output_dims = 1  if config.bid_gp else len(self.classes)

        if config.use_subset and (len(self.X_train) < config.subset_size):
            config.use_subset = 0
            config.use_validation = 0

    def set_classes(self, X, y):
        for cl in self.classes.values():
            self.data_by_classes[cl] = [i for i in range(len(X)) if y[i] == cl]


class SaveableProg:
    def __init__(self, prog):
        self.is_none = 0
        if prog is None:
            self.is_none = 1
        else:
            self.prog = prog.prog.base.tolist()
            self.class_label = prog.class_label

    def convert_to_prog(self):
        if self.is_none:
            return None

        new_prog = Prog(self.prog)
        new_prog.class_label = self.class_label
        return new_prog


class SaveableHost:
    def __init__(self, host):
        self.is_none = 0
        if host is None:
            self.is_none = 1
        else:
            self.progs_i = host.progs_i.base.tolist()

    def convert_to_host(self):
        if self.is_none:
            return None

        new_host = Host()
        new_host.set_progs(array('i', self.progs_i))
        return new_host


def load_saved(filename):
    with open(filename, 'r') as f:
        objs = f.read()
    decoded = jp.decode(objs)
    pop = [x.convert_to_prog() for x in decoded[0]]
    hosts = np.asarray([x.convert_to_host() for x in decoded[1]])
    data = decoded[2]
    env = decoded[3]
    results = decoded[4]

    env.ops = array('i', env.ops)
    data.curr_X = np.asarray(data.curr_X)
    data.curr_y = array('i', data.curr_y)
    data.curr_y = array('i', data.curr_y)
    data.curr_i = array('i', data.curr_i)
    data.load_data(env)
    return pop, hosts, data, env, results


# Initializing data
def load_data(fname, split=','):
    data = []
    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            l = line.split(split)
            if l[-1][-1] == '\n':
                l[-1] = l[-1][:-1]
            data.append(l)
    return data


def load_mnist():
    X_train = np.frombuffer(gzip.open(const.MNIST_DATA_FILES['X_train'], 'rb').read(), dtype=np.float64).reshape(60000,
                                                                                                                 784)
    X_test = np.frombuffer(gzip.open(const.MNIST_DATA_FILES['X_test'], 'rb').read(), dtype=np.float64).reshape(10000,
                                                                                                               784)
    y_train = np.frombuffer(gzip.open(const.MNIST_DATA_FILES['y_train'], 'rb').read(), dtype=np.uint8).astype(np.intc)
    y_test = np.frombuffer(gzip.open(const.MNIST_DATA_FILES['y_test'], 'rb').read(), dtype=np.uint8).astype(np.intc)
    return_vals = [X_train, y_train, X_test, y_test]
    for arr in return_vals:
        arr.flags.writeable = True
    return return_vals


def get_classes(data):
    classes = sorted(set(data))
    classmap = {}
    for i in range(len(classes)):
        cl = classes[i]
        classmap[cl] = i
    return classmap


def standardize(train, test, method, alpha=1):
    num_attrs = len(train[0])
    vals0, vals1 = [None] * num_attrs, [None] * num_attrs
    standardized = []
    matrices = [np.asmatrix(train, dtype=np.float64), np.asmatrix(test, dtype=np.float64)]
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
            standardized.append(m.tolist())
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
            standardized.append(m.tolist())
        else:
            raise AttributeError('Invalid standardize method')
    return standardized


def preprocess(data):
    for i in range(len(data)):
        try:
            data[i] = [np.float64(x) for x in data[i]]
        except ValueError:
            preprocess(convert_non_num_data(data))
    return data


def convert_non_num_data(data):
    attrs = []
    for i in range(len(data)):
        attrs += [attr for attr in data[i]]
    attrs = list(set(attrs))
    for i in range(len(data)):
        data[i] = [attrs.index(attr) + 1 for attr in data[i]]
    return data


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
            x_ind += np.random.choice(data_by_classes[i], subset_size).tolist()
        subset_y += [i] * subset_size
    subset_x = [data.X_train[i] for i in x_ind]

    if not data.act_subset_size:
        data.act_subset_size = len(subset_x)

    return np.array(subset_x), array('i', subset_y), array('i', x_ind)


def uniformprob_data_subset(data, subset_size):
    X = data.X_train
    x_ind = array('i', np.random.choice(range(len(X)), subset_size))
    subset_x = np.array([data.X_train[i] for i in x_ind])
    subset_y = array('i', [data.y_train[i] for i in x_ind])
    if not data.act_subset_size:
        data.act_subset_size = len(subset_x)
    return subset_x, subset_y, x_ind

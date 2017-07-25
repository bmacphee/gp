import const, matplotlib, gzip, pdb
import numpy as np

matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from array import array


class Point:
    def __init__(self, index, data):
        self.index = index
        self.data = data


# Initializing data
def load_data(fname, split=','):
    data = []
    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            l = line.split(split)
            if l[-1][-1] == '\n':
                l[-1] = l[-1][:-1]
            data.append(l)
            #data.append(Point(i, l))
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
            #x_ind += train_test_split(data_by_classes[i], train_size=(subset_size / class_size))[0]
        subset_y += [i] * subset_size
    subset_x = [data.X_train[i] for i in x_ind]

    if not data.act_subset_size:
        data.act_subset_size = len(subset_x)

    return np.array(subset_x), array('i', subset_y), x_ind


def uniformprob_data_subset(data, subset_size):
    X = data.X_train
    y = data.y_train
    subset_x, temp, subset_y, temp = train_test_split(X, y, train_size=(subset_size / len(X)))

    x_ind = np.random.choice(range(len(X)), subset_size)
    subset_x = np.array([data.X_train[i] for i in x_ind])
    subset_y = array('i', [data.y_train[i] for i in x_ind])
    if not data.act_subset_size:
        data.act_subset_size = len(subset_x)
    return subset_x, subset_y, x_ind



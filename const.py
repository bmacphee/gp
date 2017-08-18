from enum import Enum

TARGET = 0
SOURCE = 1
OP = 2
MODE = 3

TOURNAMENT_SIZE = 4
GEN_REGS = 8

GENREGS_MODE_VAL = 0
IP_MODE_VAL = 1


IMAGE_DIR = '/home/selene/Documents/results'
JSON_DIR = '/home/selene/Documents/results/JSON'
GRAPHINFO_DIR = '/home/selene/Documents/results/graphing/results'

TEST_DATA_FILES = {
    'data/ann-train.data': 'data/ann-test.data',
    'data/shuttle.trn': 'data/shuttle.tst',
    'data/MNIST/train-images.idx3-ubyte': 'data/MNIST/t10k-images.idx3-ubyte',
    'data/gisette_train.data': 'data/gisette_test.data'
}

LABEL_FILES = {
    'data/gisette_train.data': 'data/gisette_train.labels',
    'data/gisette_test.data': 'data/gisette_test.labels',
}
MNIST_DATA_FILES = {
    'X_train': 'data/mnist_xtrain_standardized.gz',
    'X_test': 'data/mnist_xtest_standardized.gz',
    'y_train': 'data/mnist_ytrain.gz',
    'y_test': 'data/mnist_ytest.gz',
    'all': 'data/MNIST',
}

FILE_NAMES = {
    'graph_cumulative': '_cumulative_detectrate.png',
    'graph_fitness': '_fitness.png',
    'graph_percs': '_classes.png',
    'graph_teamsize': '_team_size.png',
}

OPS = {
    0: '+',
    1: '-',
    2: '*',
    3: '/',
    4: 'sin',
    5: 'e',
    6: 'ln',
    7: 'conditional'
}


class FitnessEval(Enum):
    ACCURACY = 'accuracy'
    FITNESS_SHARING = 'fitness_sharing'
    AVG_DETECT_RATE = 'avg_detect_rate'


class StandardizeMethod(Enum):
    MEAN_VARIANCE = 0
    LINEAR_TRANSFORM = 1


class Selection(Enum):
    STEADY_STATE_TOURN = 'Steady State Tournament'
    BREEDER_MODEL = 'Breeder Model'

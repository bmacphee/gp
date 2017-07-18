from enum import Enum

TARGET = 0
SOURCE = 1
OP = 2
MODE = 3
PROG_LENGTH = 48
TOURNAMENT_SIZE = 4
GEN_REGS = 8
IMAGE_DIR = '/home/selene/Documents/results'
TEST_DATA_FILES = {
    'data/ann-train.data': 'data/ann-test.data',
    'data/shuttle.trn': 'data/shuttle.tst',
    'data/MNIST/train-images.idx3-ubyte': 'data/MNIST/t10k-images.idx3-ubyte'
}

MNIST_DATA_FILES = {
    'X_train': 'data/mnist_xtrain_standardized.gz',
    'X_test': 'data/mnist_xtest_standardized.gz',
    'y_train': 'data/mnist_ytrain.gz',
    'y_test': 'data/mnist_ytest.gz'
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
# Not used yet
#VAR_OPS = [gp.recombination, gp.mutation]

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

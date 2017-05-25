from enum import Enum

TARGET = 0
SOURCE = 1
OP = 2
MODE = 3
PROG_LENGTH = 48
TOURNAMENT_SIZE = 4
GEN_REGS = 8

TEST_DATA_FILES = { 'data/ann-train.data': 'data/ann-test.data',
                    'data/shuttle.trn': 'data/shuttle.tst' }

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
    STEADY_STATE_TOURN = 'Steady State Tournmanet'
    BREEDER_MODEL = 'Breeder Model'

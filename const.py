from enum import Enum

TARGET = 0
SOURCE = 1
OP = 2
MODE = 3
PROG_LENGTH = 48
TOURNAMENT_SIZE = 4
GEN_REGS = 8

class StandardizeMethod(Enum):
    MEAN_VARIANCE = 0
    LINEAR_TRANSFORM = 1

class Selection(Enum):
    STEADY_STATE_TOURN = 0
    BREEDER_MODEL = 1

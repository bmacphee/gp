import numpy as np
cimport numpy as np
import array
from cpython cimport array



cdef class Prog:
    cdef public np.ndarray prog
    cdef public array.array effective_instrs
    cdef array.array train_y_pred
    cdef public double trainset_trainfit, trainset_testfit, testset_testfit



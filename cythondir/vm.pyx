#!python
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import const
import numpy as np
cimport numpy as np
import array
from cpython cimport array
cimport cython
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

DTYPE_I = np.int
DTYPE_D = np.float64
ctypedef np.int_t DTYPE_I_t
ctypedef np.float64_t DTYPE_D_t

cpdef double default_val = 1.0
cpdef int num_genregs = 8
cpdef int num_ipregs
cpdef int output_dims
cpdef int prog_len = const.PROG_LENGTH
cpdef int target_i = const.TARGET, source_i = const.SOURCE, mode_i = const.MODE, op_i = const.OP

cdef extern from "math.h":
    float sqrt(float) nogil

cdef extern from "math.h":
    cpdef double sin(double x) nogil

cdef extern from "math.h":
    cpdef double log(double x) nogil

cdef extern from "math.h":
    cpdef double exp(double x) nogil

cdef extern from "math.h":
    cpdef double fabs(double x) nogil

cdef extern from "string.h":
    int strcmp(const char * lhs, const char * rhs )


def get_vals():
    return num_genregs, num_ipregs, output_dims


cdef class Point:
    cdef:
        DTYPE_D_t[:] values
        DTYPE_D fitness
    def __cinit__(self, np.ndarray values):
        self.values = values

    def fitness(self):
        pass

cdef class Prog:
    cdef public DTYPE_I_t[:,:] prog
    cdef public int[:] effective_instrs
    cdef DTYPE_I_t[:] train_y_pred

    def __cinit__(self, prog not None):
        self.prog = np.asarray(prog, dtype=DTYPE_I)
        self.effective_instrs = array.array('i', [-1])
        # Fitness for training eval on train, testing eval on train, testing eval on test

    cpdef copy(self):
        new_prog = [col[:] for col in self.prog]
        pr = Prog(new_prog)
        pr.effective_instrs = array.array('i', [-1])
        return pr

    cpdef DTYPE_I_t run_prog(self, DTYPE_D_t[:] ip) except -1:
        global num_genregs, num_ipregs, prog_len, target_i, source_i, mode_i, op_i
        cdef:
            DTYPE_I_t[:,:] pr = self.prog
            DTYPE_I_t[:] target_col = pr[target_i]
            DTYPE_I_t[:] source_col = pr[source_i]
            DTYPE_I_t[:] mode_col = pr[mode_i]
            DTYPE_I_t[:] op_col = pr[op_i]
            int max_ind = 0

        if self.effective_instrs[0] == -1:
            self.effective_instrs = findintrons(target_col, source_col, mode_col)

        cdef:
            DTYPE_I_t i, ip_len = len(ip), s = len(self.effective_instrs)
            double* output
            int[:] effective_instrs = self.effective_instrs

        with nogil:
            output = register_vals(s, effective_instrs, target_col, source_col, mode_col, op_col, ip)
        for i in range(1, output_dims):
            if output[i] > output[max_ind]:
                max_ind = i

        free(output)
        return max_ind


cpdef DTYPE_I_t[:,::1] y_pred(Prog[:] progs, DTYPE_D_t[:,:] X):
    cdef:
        int num_progs = len(progs), num_ex = len(X)
        DTYPE_I_t[:,::1] all_y_pred = np.empty((num_progs, num_ex), dtype=DTYPE_I)
        DTYPE_I_t[:] y_pred = np.empty(num_ex, dtype=DTYPE_I)
        Py_ssize_t i, j
        Prog prog
        DTYPE_D_t[:] curr_ex, curr_y_pred

    for i in range(num_progs):
        prog = progs[i]
        for j in range(num_ex):
            curr_ex = X[j]
            curr_y_pred = prog.run_prog(curr_ex)
        all_y_pred[i] = y_pred
        prog.train_y_pred = y_pred
    return all_y_pred


cpdef DTYPE_D_t avg_detect_rate(Prog prog, int[:] y, int[:] y_pred) except -1:
    cdef:
        DTYPE_I_t num_classes = max(y)+1, act_classes = len(set(y)), y_len = len(y), cl, i
        DTYPE_D_t fitness = 0.0
        double* cl_results = <double*> malloc(sizeof(double)*num_classes)
        double* percentages = <double*> malloc(sizeof(double)*num_classes)

    for i in range(num_classes):
        cl_results[i] = 0
        percentages[i] = 0

    for i in range(y_len):
        cl_results[y[i]] += 1
        if y[i] == y_pred[i]:
            percentages[y[i]] += 1

    for i in range(num_classes):
        if cl_results[i] != 0:
            fitness += (percentages[i] / cl_results[i])

    fitness = fitness/act_classes
    free(cl_results)
    free(percentages)
    return fitness


cpdef DTYPE_D_t[:] fitness_sharing(Prog[:] pop, DTYPE_D_t[:,:] X, int[:] y):
    cdef:
        Py_ssize_t i, j, k
        DTYPE_I_t len_y = len(y), len_pop = len(pop), len_row, init_fitness = 1, curr_y, last_row = 0
        DTYPE_I_t[:, :] all_y_pred = y_pred(pop, X)
        DTYPE_D_t[:] fitness = array.array('d', [0]*len_pop)
        int* denoms = <int*> malloc(sizeof(int)*len_y)
        int* row = <int*> malloc(sizeof(int)*len_pop*len_y)
        int* col = <int*> malloc(sizeof(int)*len_pop*len_y)

    for i in range(len_y):
        denoms[i] = 1
        curr_y = y[i]
        for j in range(len_pop):
            if all_y_pred[j, i] == curr_y:
                denoms[i] += 1
                row[last_row] = j
                col[last_row] = i
                last_row += 1

    for i in range(last_row):
        j = row[i]
        k = col[i]
        fitness[j] += (1.0 / denoms[k])

    free(denoms)
    free(row)
    free(col)
    return fitness

cpdef void init(int genregs, int ipregs, int outdims):
    global num_genregs, num_ipregs, output_dims

    num_genregs = genregs
    num_ipregs = ipregs
    output_dims = outdims


cdef double* register_vals(DTYPE_I_t length, int[:] effective_instrs, DTYPE_I_t[:] target_col, DTYPE_I_t[:] source_col, DTYPE_I_t[:] mode_col, DTYPE_I_t[:] op_col, DTYPE_D_t[:] ip) nogil:
    global num_genregs, default_val
    cdef char* n = '\n'
    cdef:
        DTYPE_I_t i, j, mode, op_code, target, source_val
        double source
        double* gen_regs = <double*> malloc(sizeof(double)*num_genregs)
    for i in range(num_genregs):
        gen_regs[i] = default_val

    for j in range(length):
        i = effective_instrs[j]
        mode = mode_col[i]
        op_code = op_col[i]
        target = target_col[i]
        source_val = source_col[i]
        if mode == 0:
            source = gen_regs[source_val % num_genregs]
        else:
            source = ip[source_val % num_ipregs]
        gen_regs[target] = calc(source, gen_regs[target], op_code)
    return gen_regs


cdef double calc(double source, double target, double op_code) nogil:
    cdef double penalty = -100000
    if op_code == 0:
        return target+source
    elif op_code == 1:
        return target-source
    elif op_code == 2:
        return target*source
    elif op_code == 3:
        if source != 0:
            return target/source
        else:
            return 1
    elif op_code == 4:
        return sin(source)
    elif op_code == 5:
        if fabs(source) <= 32:
            # Penalty?
            return exp(source)
        else:
            return 1
    elif op_code == 6:
        if source != 0:
            return log(fabs(source))
        else:
            return 1


cpdef array.array[int] find_introns(DTYPE_I_t[:] target,  DTYPE_I_t[:] source, DTYPE_I_t[:] mode):
    global num_genregs, output_dims

    cdef:
        list marked_instrs = []
        set eff_regs = set(range(output_dims))
        int m, s, i = len(target)-1

    while i >= 0:
        if target[i] in eff_regs:
            marked_instrs.insert(0, i)
            m = mode[i]
            # Mode = 0: source is a general register
            # TODO: effective regs for single-op instructions?
            if mode[i] == 0:
                s = source[i]
                eff_regs.add(s % num_genregs)
        i -= 1
    return array.array('i', marked_instrs)


cpdef array.array[int] findintrons(DTYPE_I_t[:] target,  DTYPE_I_t[:] source, DTYPE_I_t[:] mode):
    global num_genregs, output_dims

    cdef:
        list marked_instrs = []
        set eff_regs = set(range(output_dims))
        int m, s, i = len(target)-1

    while i >= 0:
        if target[i] in eff_regs:
            marked_instrs.insert(0, i)
            m = mode[i]
            # Mode = 0: source is a general register
            # TODO: effective regs for single-op instructions?
            if mode[i] == 0:
                s = source[i]
                eff_regs.add(s % num_genregs)
        i -= 1
    return array.array('i', marked_instrs)
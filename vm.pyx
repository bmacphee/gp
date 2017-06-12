# cython: profile=True
import const
import numpy as np
cimport numpy as np

import array
from cpython cimport array
#from libc.math cimport sqrt
cimport cython
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

from libc.string cimport memset



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

cpdef double default_val = 1
cpdef int num_genregs = 8
cpdef int num_ipregs
cpdef int output_dims
cpdef int prog_len = const.PROG_LENGTH
cpdef int target_i = const.TARGET, source_i = const.SOURCE, mode_i = const.MODE, op_i = const.OP

DTYPE_I = np.int
ctypedef np.int_t DTYPE_I_t
DTYPE_D = np.float64
ctypedef np.float64_t DTYPE_D_t

def get_vals():
    return num_genregs, num_ipregs, output_dims

cdef class Prog:
    cdef public list prog
    cdef public array.array effective_instrs
    cdef array.array train_y_pred
    cdef public double trainset_trainfit, trainset_testfit, testset_testfit
    def __init__(self, prog):
        self.prog = prog
        self.effective_instrs = array.array('i', [-1])
        # Fitness for training eval on train, testing eval on train, testing eval on test
        self.trainset_trainfit = -1
        self.trainset_testfit = -1
        self.testset_testfit = -1
        self.train_y_pred = array.array('i', [-1])

    def copy(self):
        new_prog = [col[:] for col in self.prog]
        pr = Prog(new_prog)
        pr.trainset_trainfit = self.trainset_trainfit
        pr.trainset_testfit = self.trainset_testfit
        pr.testset_testfit = self.testset_testfit
        pr.effective_instrs = self.effective_instrs[:] if self.effective_instrs else None
        pr.train_y_pred = self.train_y_pred[:] if self.train_y_pred else None
        return pr


cdef class Vm:
    cdef public int num_genregs, num_ipregs, output_dims, prog_len
    cdef public np.ndarray X
    cdef double default_val

    def __init__(self, num_genregs, num_ipregs, output_dims, prog_len):
        self.num_genregs = num_genregs
        self.num_ipregs = num_ipregs
        self.output_dims = output_dims
        self.prog_len = prog_len
        self.default_val = 1.0

    cpdef np.ndarray run(self, list progs, np.ndarray X, int fitness_sharing):
        print('START')
        cdef int num_progs = len(progs)

        cdef Prog prog
        cdef np.ndarray ex
        cdef int cl
        cdef int X_len = len(X[0])

        cdef np.ndarray all_y_pred = np.empty((num_progs, X_len), dtype=np.int32)
        #cdef int* y_pred = <int *> malloc(sizeof(int) * X_len)
        cdef np.ndarray y_pred = np.empty((1, X_len), dtype=np.int32)


        cdef int i, j
        for i in range(num_progs):
            for j in range(X_len):
                y_pred[0][j] = run_prog(progs[i], X[j])
            all_y_pred[i] = np.ndarray([y_pred])
        return all_y_pred

    cpdef void set_X(self, np.ndarray X):
        self.X = X



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list y_pred(list progs, np.ndarray X):
    cdef int num_progs = len(progs), num_ex = len(X)
    #cdef Prog* proglist = <Prog*> malloc(sizeof(Prog)*num_progs)
    cdef np.ndarray all_y_pred = np.empty((num_progs, num_ex), dtype=DTYPE_I)
    cdef int[:] y_pred = array.array('i', range(num_ex))
    cdef int i, j
    cdef Prog prog

    for i in range(num_progs):
        prog = progs[i]
        # train_y_pred = prog.train_y_pred
        # if set_train_pred and (train_y_pred[0] != -1):
        #     all_y_pred[i] = train_y_pred
        # else:
        for j in range(num_ex):
            y_pred[j] = run_prog(prog, X[j])
        all_y_pred[i] = y_pred
    return all_y_pred.tolist()



@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_I_t[:, :] ypred(list progs, np.ndarray X):
    cdef int num_progs = len(progs), num_ex = len(X)
    #cdef Prog* proglist = <Prog*> malloc(sizeof(Prog)*num_progs)
    #cdef np.ndarray all_y_pred = np.empty((num_progs, num_ex), dtype=DTYPE_I_t)
    cdef np.ndarray all_y_pred = np.empty((num_progs, num_ex), dtype=DTYPE_I)
    #cdef array.array[int] y_pred = array.array('i', [-1]*num_ex)
    cdef np.ndarray y_pred = np.empty(num_ex, dtype=DTYPE_I)
    cdef int i, j
    cdef Prog prog

    for i in range(num_progs):
        prog = progs[i]
        for j in range(num_ex):
            y_pred[j] = run_prog(prog, X[j])
        all_y_pred[i] = y_pred
    #cdef DTYPE_I_t[:,:] all_y_pred_=all_y_pred
    return all_y_pred


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DTYPE_D_t avg_detect_rate(Prog prog, int[:] y, int[:] y_pred) except -1:
    cdef:
        int num_classes = len(set(y)), y_len = len(y), cl, i
        #DTYPE_D_t[::1] percentages = np.zeros(num_classes, dtype=DTYPE_D), cl_results = np.zeros(num_classes, dtype=DTYPE_D)
        DTYPE_D_t fitness = 0.0

        double* cl_results = <double*> malloc(sizeof(double)*num_classes)
        double* percentages = <double*> malloc(sizeof(double)*num_classes)


    for i in range(y_len):
        cl_results[y[i]] += 1
        if y[i] == y_pred[i]:
            percentages[y[i]] += 1

    for i in range(num_classes):
        fitness += (percentages[i] / cl_results[i])

    fitness = fitness/num_classes
    free(cl_results)
    free(percentages)
    # if store_fitness:
    #     setattr(prog, store_fitness, fitness)
    return fitness

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray fitness_sharing(list pop, np.ndarray X, int[:] y):
    cdef:
        Py_ssize_t i, j, k
        DTYPE_I_t len_y = len(y), len_pop = len(pop), len_row, init_fitness = 1, curr_y, last_row = 0
        DTYPE_I_t[:, :] all_y_pred = ypred(pop, X)
        #DTYPE_I_t[:] denoms = np.empty(len_y, dtype=DTYPE_I)
        np.ndarray fitness = np.empty(len_pop, dtype=DTYPE_D)
        int* denoms = <int*> malloc(sizeof(int)*len_y)
        int* row = <int*> malloc(sizeof(int)*len_pop*len_y)
        int* col = <int*> malloc(sizeof(int)*len_pop*len_y)
        #double* fitness = <double*> malloc(sizeof(double)*len_pop)
        #DTYPE_I_t[:,:] ypred_transpose = all_y_pred.T
        ##DTYPE_I_t[:] curr_col
        #np.ndarray curr_y = np.empty(1, dtype=int)


    for i in range(len_y):
        denoms[i] = 1
        curr_y = y[i]
        #curr_col = all_y_pred[:,i]
        #col = ypred_transpose[i]
        #curr_col = ypred_transpose[i]
        #denoms[i] += np.sum(col == curr_y) + 1
        for j in range(len_pop):
            if init_fitness == 1:
                fitness[j] = 0
            if all_y_pred[i, j] == curr_y:
                denoms[i] += 1
                row[last_row] = j
                col[last_row] = i
                last_row += 1
        if init_fitness == 1:
            init_fitness = 0

    #row, col = np.where(all_y_pred.base == y)
    # len_row = row.size
    for i in range(last_row):
        j = row[i]
        k = col[i]
        fitness[j] += (1.0 / denoms[k])
    # denoms = [sum([1 for j in range(len(all_y_pred)) if all_y_pred[j][i] == y[i]]) + 1 for i in range(len(y))]
    # fitness = [sum([int(y_pred[i] == y[i]) / denoms[i] for i in range(len(y))]) for y_pred in all_y_pred]
    #return_fitness = [fitness[i] for i in range(len_pop)]

    free(denoms)
    #free(fitness)
    free(row)
    free(col)
    return fitness

cpdef void init(int genregs, int ipregs, int outdims):
    global num_genregs, num_ipregs, output_dims

    num_genregs = genregs
    num_ipregs = ipregs
    output_dims = outdims



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
#@cython.locals(target_col=cython.int[48], source_col=cython.int[48], mode_col=cython.int[48], op_col=cython.int[48])
cpdef DTYPE_I_t run_prog(Prog prog, np.ndarray ip) except -1:
#cdef int run_prog(int[:] target_col, int[:] source_col, int[:] mode_col, int[:] op_col, np.ndarray ip):
    global num_genregs, num_ipregs, prog_len, target_i, source_i, mode_i, op_i
    #
    cdef:
        int[:] target_col = prog.prog[target_i]
        int[:] source_col = prog.prog[source_i]
        int[:] mode_col = prog.prog[mode_i]
        int[:] op_col = prog.prog[op_i]
    #
    if prog.effective_instrs is None:
        setattr(prog, 'effective_instrs', find_introns(target_col, source_col, mode_col))

    cdef:
        int i, ip_len = len(ip), s = len(prog.effective_instrs)
        double* output
        DTYPE_D_t[:] ip_ = ip
        #int* effective_instrs = <int*> malloc(sizeof(int)*s)
        int[:] effective_instrs = prog.effective_instrs

    # for i in range(s):
    #     effective_instrs[i] = prog.effective_instrs[i]

    with nogil:
        output = runprog(s, effective_instrs, target_col, source_col, mode_col, op_col, ip_)
        #free(effective_instrs)
        # free(ip_)

    #list out = [output[x] for x in range(output_dims)]
    cdef int max_ind

    max_ind = 0
    for i in range(1, output_dims):
        if output[i] > output[max_ind]:
            max_ind = i

    free(output)
    return max_ind




@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double* runprog(int length, int[:] effective_instrs, int[:] target_col, int[:] source_col, int[:] mode_col, int[:] op_col, DTYPE_D_t[:] ip) nogil:
    global num_genregs, default_val
    cdef char* n = '\n'
    cdef:
        int i, j, mode, op_code, target, source_val
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


@cython.cdivision(True)
cdef double calc(double source, double target, double op_code) nogil:
    #do_op(op_col[i], gen_regs, target_col[i], source)
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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef array.array[int] find_introns(int[:] target,  int[:] source, int[:] mode):
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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef array.array[int] findintrons(int[] target,  int[] source, int[] mode):
    global num_genregs, output_dims

    cdef:
        list marked_instrs = []
        set eff_regs = set(range(output_dims))
        int m, s, i = sizeof(target) / sizeof(int)

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

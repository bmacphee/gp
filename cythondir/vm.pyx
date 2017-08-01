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


import random

DTYPE_I = np.int
DTYPE_D = np.float64
ctypedef np.int_t DTYPE_I_t
ctypedef np.float64_t DTYPE_D_t

cpdef double default_val = 1.0
cpdef int num_genregs = 8
cpdef int num_ipregs
cpdef DTYPE_I_t output_dims
cpdef int prog_len = const.PROG_LENGTH
cpdef int target_i = const.TARGET, source_i = const.SOURCE, mode_i = const.MODE, op_i = const.OP
cpdef int bid_gp = 0
cpdef int train_size = 0
cpdef int id_num = 0

cpdef DTYPE_I_t[:,::1] curr_ypred_state = None

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
    return num_genregs, num_ipregs, output_dims, bid_gp


def get_curr_ypred():
    return curr_ypred_state

def set_curr_ypred(ypred):
    global curr_ypred_state
    curr_ypred_state = ypred


cdef class Host:
    cdef:
        public int[:] progs_i
        #public array.array progs_i
        int num_progs
        array.array active_progs

    def __init__(self):
        self.clear_progs()

    cpdef set_progs(self, array.array progs):
        self.progs_i = progs
        self.num_progs = len(progs)

    # Does not protect from duplicates
    cpdef void add_progs(self, array.array prog_i):
        cdef array.array curr_progs = self.progs_i.base[:]
        curr_progs.extend(prog_i)
        self.set_progs(curr_progs)

    cpdef void del_prog(self, int prog_i):
        self.progs_i.base.remove(prog_i)
        self.num_progs -= 1

    cpdef void clear_progs(self):
        self.progs_i = array.array('i')
        self.num_progs = 0
        self.active_progs = array.array('i')

    cpdef array.array get_progs(self):
        return self.progs_i.base

    cpdef clear_inactive(self, int[:] pop_ids):
        cdef:
            int i, active
            array.array new_progs = self.progs_i.base[:]

        for i in range(len(self.progs_i)):
            if pop_ids[self.progs_i[i]] not in self.active_progs:
                new_progs.remove(self.progs_i[i])
        self.set_progs(new_progs)

    cpdef get_active(self):
        return self.active_progs

    cpdef set_active(self, array.array active):
        self.active_progs = active

    cpdef copy(self):
        h = Host()
        h.set_progs(self.progs_i.base)
        return h




cdef class Prog:
    cdef public DTYPE_I_t[:,::1] prog
    cdef public int class_label
    cdef public DTYPE_D_t[:,::1] bid_vals
    cdef public DTYPE_D_t[:] first_50_regs
    cdef public int prog_id

    cdef int[:] effective_instrs
    cdef DTYPE_I_t[:] test_y_pred
    cdef DTYPE_D_t* output_registers
    cdef int output_dims
    cdef int ex_num

    def __cinit__(self, list prog):
        global output_dims, train_size
        self.output_registers = <DTYPE_D_t*> malloc(sizeof(DTYPE_D_t)*output_dims)
        self.output_dims = output_dims
        if not self.output_registers:
            raise MemoryError()


    def __init__(self, list prog):
        global train_size, id_num
        self.prog = np.asarray(prog, dtype=DTYPE_I)
        self.effective_instrs = array.array('i')[:]
        self.class_label = -1
        self.ex_num = -1
        self.prog_id = id_num
        id_num += 1

        self.bid_vals = np.zeros((train_size, 2))
        self.first_50_regs = np.asarray([-1]*50, dtype=DTYPE_D)

    def __dealloc__(self):
        free(self.output_registers)

    cpdef Prog copy(self):
        cdef:
            list new_prog
        new_prog = [col[:] for col in self.prog]
        pr = Prog(new_prog)
        pr.effective_instrs = array.array('i')
        pr.class_label = self.class_label
        return pr

    cpdef void run_prog(self, DTYPE_D_t[:] ip):
        global num_genregs, num_ipregs, prog_len, target_i, source_i, mode_i, op_i, bid_gp
        cdef:
            DTYPE_I_t[::1] target_col = self.prog[target_i]
            DTYPE_I_t[::1] source_col = self.prog[source_i]
            DTYPE_I_t[::1] mode_col = self.prog[mode_i]
            DTYPE_I_t[::1] op_col = self.prog[op_i]

        if self.effective_instrs.size == 0:
            self.effective_instrs = findintrons(target_col, source_col, mode_col)
        cdef DTYPE_I_t i, ip_len = len(ip), s = len(self.effective_instrs)

        with nogil:
            output = register_vals(s, self.effective_instrs, target_col, source_col, mode_col, op_col, ip)
            for i in range(self.output_dims):
                self.output_registers[i] = output[i]
        free(output)

    cpdef DTYPE_I_t max_ind(self):
        cdef DTYPE_I_t max_ind = 0, i

        for i in range(1, self.output_dims):
            if self.output_registers[i] > self.output_registers[max_ind]:
                max_ind = i
        return max_ind

    cpdef void clear_effective_instrs(self):
        self.effective_instrs = array.array('i')

    cpdef list get_regs(self):
         global num_genregs
         x = [self.output_registers[i] for i in range(output_dims)]
         return x

    cpdef int[:] test(self):
        return self.effective_instrs

    cpdef int is_duplicate(self, Prog[:] pop):
        cdef:
            DTYPE_D_t[:] vals
            Prog prog
            DTYPE_D_t diff = 0.001
            DTYPE_I_t i, j, l, c, num_progs = pop.shape[0]

        l = pop[0].first_50_regs.size
        for i in range(num_progs):
            c = 0
            prog = pop[i]
            vals = prog.first_50_regs
            for j in range(l):
                if fabs(vals[j] - self.first_50_regs[j]) <= diff:
                    c += 1
            if c == l:
                return 1
        return 0


cpdef DTYPE_I_t[:,::1] y_pred(Prog[:] progs, DTYPE_D_t[:,:] X):
    cdef:
        DTYPE_I_t curr_y_pred, num_progs = len(progs), num_ex = len(X)
        DTYPE_I_t[:,::1] all_y_pred = np.empty((num_progs, num_ex), dtype=DTYPE_I)
        DTYPE_I_t[:] y_pred = np.empty(num_ex, dtype=DTYPE_I)
        Py_ssize_t i, j
        Prog prog
        DTYPE_D_t[:] curr_ex


    for i in range(num_progs):
        prog = progs[i]
        for j in range(num_ex):
            curr_ex = X[j]
            if prog.class_label == -1:
                prog.run_prog(curr_ex)
                curr_y_pred = prog.max_ind()
            else:
                curr_y_pred = prog.class_label
            y_pred[j] = curr_y_pred
        all_y_pred[i] = y_pred
    return all_y_pred


cpdef np.ndarray host_y_pred(Prog[:] pop, Host[:] hosts, DTYPE_D_t[:,::1] X, int[:] x_inds, int change_regs):
    cdef:
        DTYPE_I_t curr_y_pred, num_hosts = len(hosts), num_ex = len(X), x_ind
        DTYPE_I_t[:,::1] all_y_pred = np.empty((num_hosts, num_ex), dtype=DTYPE_I)
        DTYPE_I_t[:] y_pred = np.empty(num_hosts, dtype=DTYPE_I)
        #size_t i, j, max_ind
        Host host
        int[:] progs_i
        Prog prog
        DTYPE_D_t[:] curr_ex
        DTYPE_D_t max_val, val
        int k, prog_i, i, j, max_ind = 0, host_ind, use_saved

    if x_inds is None:
        use_saved = 0
    else:
        use_saved = 1

    for i in range(num_ex):
        curr_ex = X[i]
        for j in range(num_hosts):
            host = hosts[j]
            progs_i = host.progs_i
            num_progs = host.num_progs

            for k in range(num_progs):
                prog_i = progs_i[k]
                prog = pop[prog_i]

                if use_saved:
                    x_ind = x_inds[i]
                    if prog.bid_vals[x_ind, 0] == 0:
                        prog.run_prog(curr_ex)
                        prog.bid_vals[x_ind, 0] = 1
                        prog.bid_vals[x_ind, 1] = prog.output_registers[0]

                    val = prog.bid_vals[x_ind,1]

                else:
                    if (prog.ex_num != i):
                        prog.run_prog(curr_ex)
                        prog.ex_num = i
                    val = prog.output_registers[0]

                if change_regs and (i < 50):
                    prog.first_50_regs[i] = val

                if k == 0:
                    max_val = val
                    max_ind = prog_i
                elif val > max_val:
                    max_val = val
                    max_ind = prog_i

            prog = pop[max_ind]
            if prog.prog_id not in host.active_progs:
                host.active_progs.insert(0, prog.prog_id)
            curr_y_pred = prog.class_label
            y_pred[j] = curr_y_pred
        all_y_pred[:, i] = y_pred
    return all_y_pred.base


# Needed to take prog argument previously (when storing) - fix this (may be called w/ a host)
cpdef DTYPE_D_t avg_detect_rate(prog, int[:] y, int[:] y_pred) except -1:
    cdef:
        DTYPE_I_t num_classes = max(y)+1, act_classes = len(set(y)), y_len = len(y), cl, i
        DTYPE_D_t fitness = 0.0
        DTYPE_D_t* cl_results = <DTYPE_D_t*> malloc(sizeof(DTYPE_D_t)*num_classes)
        DTYPE_D_t* percentages = <DTYPE_D_t*> malloc(sizeof(DTYPE_D_t)*num_classes)

    if num_classes == 0:
        return 0

    if not cl_results or not percentages:
        raise MemoryError()

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


cpdef array.array fitness_sharing(Prog[:] pop, DTYPE_D_t[:,::1] X, int[:] y, Host[:] hosts, int[:] x_inds):
    global curr_ypred_state
    cdef:
        size_t i, j, k
        DTYPE_I_t len_y = len(y), len_pop, len_row, init_fitness = 1, curr_y, last_row = 0
        DTYPE_I_t[:, ::1] all_y_pred
        array.array fitness
        DTYPE_D_t* denoms = <DTYPE_D_t*> malloc(sizeof(DTYPE_D_t)*len_y)
        DTYPE_I_t* row
        DTYPE_I_t* col
        DTYPE_D_t numer = 1.0

    if hosts is None:
        all_y_pred = y_pred(pop, X)
        len_pop = len(pop)
    else:
        all_y_pred = host_y_pred(pop, hosts, X, x_inds, 1)
        len_pop = len(hosts)

    curr_ypred_state = all_y_pred
    row = < DTYPE_I_t * > malloc(sizeof(DTYPE_I_t) * len_pop * len_y)
    col = < DTYPE_I_t * > malloc(sizeof(DTYPE_I_t) * len_pop * len_y)
    fitness = array.array('d', [0.0]*len_pop)

    if not denoms or not row or not col:
        raise MemoryError()

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
        fitness[j] += (numer / denoms[k])

    free(denoms)
    free(row)
    free(col)
    return fitness


cpdef DTYPE_D_t[:] point_fitness(array.array[int] curr_y):
    #fit = 1+(1-c_k)/M_size if c_k > 0 else 0
    global curr_ypred_state
    assert curr_ypred_state is not None, 'Fitness evaluation should be run first'

    cdef:
        DTYPE_I_t[::1,:] ypred_t = curr_ypred_state.T
        DTYPE_I_t row_len = ypred_t.shape[0], col_len = ypred_t.shape[1]
        DTYPE_I_t i, j
        DTYPE_D_t msize = ypred_t.shape[1], corr
        DTYPE_D_t[:] fitness = np.empty(row_len, dtype=DTYPE_D)

    for i in range(row_len):
        corr = 0
        for j in range(col_len):
            if ypred_t[i,j] == curr_y[i]:
                corr += 1
        if corr > 0:
            fitness[i] = (1+((1-corr)/col_len))
        else:
            fitness[i] = 0

    return fitness


cpdef void init(int genregs, int ipregs, int outdims, int bid, int trainsize):
    global num_genregs, num_ipregs, output_dims, bid_gp, train_size

    num_genregs = genregs
    num_ipregs = ipregs
    output_dims = outdims
    bid_gp = bid
    train_size = trainsize

cdef DTYPE_D_t* register_vals(DTYPE_I_t length, int[:] effective_instrs, DTYPE_I_t[::1] target_col, DTYPE_I_t[::1] source_col, DTYPE_I_t[::1] mode_col, DTYPE_I_t[::1] op_col, DTYPE_D_t[:] ip) nogil:
    global num_genregs, default_val

    cdef:
        DTYPE_I_t i, j, mode, op_code, target, source_val
        DTYPE_D_t source
        DTYPE_D_t* gen_regs = <DTYPE_D_t*> malloc(sizeof(DTYPE_D_t)*num_genregs)
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


cdef DTYPE_D_t calc(double source, double target, double op_code) nogil:
    cdef DTYPE_D_t penalty = -100000
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
            #return penalty
            return 1
    elif op_code == 4:
        return sin(source)
    elif op_code == 5:
        if fabs(source) <= 32:
            # Penalty?
            return exp(source)
        else:
            return 1
            # return penalty
    elif op_code == 6:
        if source != 0:
            return log(fabs(source))
        else:
            return 1
            # return penalty
    elif op_code == 7:
        if target < source:
            return -(target)
        else:
            return target


cpdef void set_introns(Prog prog):
    global target_i, source_i, mode_i

    cdef:
        DTYPE_I_t[:] target = prog.prog[target_i]
        DTYPE_I_t[:] source = prog.prog[source_i]
        DTYPE_I_t[:] mode = prog.prog[mode_i]

    prog.effective_instrs = findintrons(target, source, mode)


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
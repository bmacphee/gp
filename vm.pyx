
import const
import numpy as np
cimport numpy as np
from cpython cimport array
import array
#from libc.math cimport sqrt
cimport cython

from libc.stdlib cimport malloc, free
from cpython cimport array
from libc.stdio cimport printf

cdef extern from "math.h":
    float sqrt(float) nogil

# cpdef class Prog:
#     def __init__(self, prog):
#         self.prog = prog
#         self.effective_instrs = None
#         self.fitness = None
#         self.acc_fitness = None
#         self.train_y_pred = None

# default_val = float(1)
# cdef int num_genregs = 0
# cdef int num_ipregs = 0
# cdef int num_ops = 0
# cdef int output_dims = 0
# cdef int prog_len = const.PROG_LENGTH
# test = num_genregs
cdef class Vm(object):
    cdef public int num_genregs, num_ipregs, output_dims
    cdef int prog_len
    cdef float default_val

    def __init__(self, num_genregs, num_ipregs, output_dims):
        self.num_genregs = num_genregs
        self.num_ipregs = num_ipregs
        self.output_dims = output_dims

cdef extern from "math.h":
    cpdef double sin(double x) nogil

cdef extern from "math.h":
    cpdef double pow(double x, double y) nogil

cdef extern from "math.h":
    cpdef double fabs(double x) nogil

cdef double default_val = 1
cdef int num_genregs = 8
cdef int num_ipregs
cdef int num_ops
cdef int output_dims
cdef int prog_len = const.PROG_LENGTH

def get_vals():
    return num_genregs, num_ipregs, num_ops, output_dims

cpdef void init(int genregs, int ipregs, int outdims):
    global num_genregs, num_ipregs, output_dims

    num_genregs = genregs
    num_ipregs = ipregs
    output_dims = outdims

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(target_col=cython.int[48], source_col=cython.int[48], mode_col=cython.int[48], op_col=cython.int[48])
cpdef list run_prog(prog, array.array[double] ip):
    global num_genregs, num_ipregs, prog_len

    # cdef int[48] target_col = array.array('i', prog.prog[const.TARGET])
    # cdef int[48] source_col = array.array('i', prog.prog[const.SOURCE])
    # cdef int[48] mode_col = array.array('i', prog.prog[const.MODE])
    # cdef int[48] op_col = array.array('i', prog.prog[const.OP])
    target_col = prog.prog[const.TARGET]
    source_col = prog.prog[const.SOURCE]
    mode_col = prog.prog[const.MODE]
    op_col = prog.prog[const.OP]

    if prog.effective_instrs is None:
        setattr(prog, 'effective_instrs', find_introns(target_col, source_col, mode_col))

    cdef int s = len(prog.effective_instrs)
    cdef int* effective_instrs = <int*> malloc(sizeof(int)*s)
    cdef int i
    for i in range(s):
        effective_instrs[i] = prog.effective_instrs[i]

    cdef int ip_len = len(ip)
    cdef double* ip_
    cdef double* output
    with nogil:
        ip_ = <double*> malloc(sizeof(double)*ip_len)
        for i in range(ip_len):
            ip_[i] = ip[i]
        output = runprog(s, effective_instrs, target_col, source_col, mode_col, op_col, ip_)
        free(effective_instrs)
        free(ip_)
    # for i in range(8):
    #     printf('%f\t',  output[i])
    # printf('\n')
    out = [output[x] for x in range(output_dims)]

    free(output)
    return out

@cython.cdivision(True)
cdef double* runprog(int length, int[] effective_instrs, int[] target_col, int[] source_col, int[] mode_col, int[] op_col, double[] ip) nogil:
    global num_genregs, default_val

    cdef int i
    #cdef double[8] gen_regs
    cdef double* gen_regs = <double*> malloc(sizeof(double)*num_genregs)
    for i in range(num_genregs):
        gen_regs[i] = default_val
    # mode = {
    #     0: (lambda x:gen_regs[x % num_genregs]),
    #     1: (lambda x:ip_regs[x % num_ipregs])
    # }
    #mode = [(lambda x:gen_regs[x % num_genregs]),(lambda x:ip_regs[x % num_ipregs])]


    # cdef int[:] target_col = array.array('i', prog.prog[const.TARGET])[:]
    # cdef int[:] source_col = array.array('i', prog.prog[const.SOURCE])[:]
    # cdef int[:] mode_col = array.array('i', prog.prog[const.MODE])[:]
    # cdef int[:] op_col = array.array('i', prog.prog[const.OP])[:]

    cdef int j, mode, op_code, target, source_val
    #cdef int l = effective_instrs.size
    cdef double source
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
cdef inline double calc(double source, double target, double op_code) nogil:
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
            return target + penalty
    elif op_code == 4:
        return sin(target)
    elif op_code == 5:
        if fabs(source) <= 10:
            return pow(fabs(target), source)
        else:
            return target + penalty



# Return set w/ indices of effective instructions
cpdef find_introns(list target, list source, list mode):
    return introns(target, source, mode)

cdef list introns(list target, list source, list mode):
    global num_genregs, output_dims

    cdef list marked_instrs = []
    cdef set eff_regs = set(range(output_dims))
    cdef int i = len(target)-1
    cdef int m, s

    while i >= 0:
        if target[i] in eff_regs:
            marked_instrs.insert(0, i)
            m = mode[i]
            # Mode = 0: source is a general register
            # TODO: effective regs shouldn't include source for single-op instructions
            if mode[i] == 0:
                s = source[i]
                eff_regs.add(s % num_genregs)
        i -= 1
    return marked_instrs

#cpdef list get_predicted_classes(list progs, list X, int fitness_sharing):

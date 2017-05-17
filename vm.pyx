import const
import numpy as np
cimport numpy as np
from cpython cimport array
import array
from libc.math cimport sqrt

default_val = float(1)
num_genregs = 0
num_ipregs = 0
num_ops = 0
output_dims = 0
prog_len = const.PROG_LENGTH

def run_prog(prog, list ip):
    return runprog(prog, ip)

cdef list runprog(prog, list ip):
    cdef list ip_regs = [float(x) for x in ip]
    cdef list gen_regs = [default_val]*num_genregs

    # mode = {
    #     0: (lambda x:gen_regs[x % num_genregs]),
    #     1: (lambda x:ip_regs[x % num_ipregs])
    # }
    #mode = [(lambda x:gen_regs[x % num_genregs]),(lambda x:ip_regs[x % num_ipregs])]


    # cdef int[:] target_col = array.array('i', prog.prog[const.TARGET])[:]
    # cdef int[:] source_col = array.array('i', prog.prog[const.SOURCE])[:]
    # cdef int[:] mode_col = array.array('i', prog.prog[const.MODE])[:]
    # cdef int[:] op_col = array.array('i', prog.prog[const.OP])[:]
    cdef list target_col = prog.prog[const.TARGET]
    cdef list source_col = prog.prog[const.SOURCE]
    cdef list mode_col = prog.prog[const.MODE]
    cdef list op_col = prog.prog[const.OP]


    if not prog.effective_instrs:
        prog.effective_instrs = find_introns(target_col, source_col, mode_col)

    cdef int i, j, mode, op_code, target, source_val
    cdef np.float64_t source
    cdef list effective_instrs = prog.effective_instrs
    for j in range(len(effective_instrs)):
        i = effective_instrs[j]
        mode = mode_col[i]
        op_code = op_col[i]
        target = target_col[i]
        source_val = source_col[i]
        if mode == 0:
            source = gen_regs[source_val % num_genregs]
        else:
            source = ip_regs[source_val % num_ipregs]
        gen_regs[target] = calc(source, gen_regs[target], op_code)

    output = gen_regs[:output_dims]
    return output

cdef inline float calc(float source, float target, int op_code):
    #do_op(op_col[i], gen_regs, target_col[i], source)
    if op_code == 0:
        return target+source
    elif op_code == 1:
        return target-source
    elif op_code == 2:
        return target*source
    elif op_code == 3:
        return target/(sqrt(1+source**2))


# Return set w/ indices of effective instructions
def find_introns(list target, list source, list mode):
    # marked_instrs = set()
    # eff_regs = set(range(output_dims))
    # instrs = reversed(range(len(target)))
    # for i in instrs:
    #     if target[i] in eff_regs:
    #         marked_instrs.add(i)
    #         if mode[i] == 0:
    #             eff_regs.add(source[i] % num_genregs)
    # return sorted(marked_instrs)
    return introns(target, source, mode)

cdef list introns(list target, list source, list mode):
    cdef list marked_instrs = []
    cdef set eff_regs = set(range(output_dims))
    #cdef list instrs = reversed(range(len(target)))

    cdef int i = len(target)-1
    cdef int m, s
    while i >= 0:
        if target[i] in eff_regs:
            marked_instrs.insert(0, i)
            m = mode[i]
            if mode[i] == 0:
                s = source[i]
                eff_regs.add(s % num_genregs)
        i -= 1
    #return sorted(marked_instrs)
    return marked_instrs

cdef add(list target, int index, float source):
    target[index] += source

cdef sub(list target, int index, float source):
    target[index] -= source

cdef mult(list target, int index, float source):
    target[index] *= source

cdef div(list target, int index, float source):
    target[index] /= (sqrt(1+source**2))

#ops = { 0: add, 1: sub, 2: mult, 3: div }
ops = [ add, sub, mult, div ]


#cdef do_op(int op_code, list target, int index, float source):
cdef do_op(int op_code, float target, float source):
    # try:
    #     ops[op_code](target, index, source)
    # except Exception as e:
    #     print(e)

    #ops[op_code](target, index, source)
    # if op_code == 0:
    #     target[index] += source
    # elif op_code == 1:
    #     target[index] -= source
    # elif op_code == 2:
    #     target[index] *= source
    # elif op_code == 3:
    #     target[index] /= (sqrt(1+source**2))
    pass

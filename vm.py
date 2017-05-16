import random, pdb, const
import numpy as np
from functools import lru_cache
from math import sqrt

class Vm:
    def __init__(self, num_genregs, num_ipregs, num_ops, output_dims):
        self.default_val = 1
        self.num_genregs = num_genregs
        self.num_ipregs = num_ipregs
        self.num_ops = num_ops
        self.output_dims = output_dims
        #self.gen_regs = [self.default_val]*num_genregs
        self.prog_len = const.PROG_LENGTH
        # mode=0: general regs, mode=1: ip regs

    def run_prog(self, prog, ip, remove_introns=True):
        #self.prog_len = len(prog[0])
        #self.ip_regs = [np.float64(x) for x in ip]

        ip_regs = [np.float64(x) for x in ip]
        gen_regs = [self.default_val]*self.num_genregs
        mode = {
            0: (lambda x:gen_regs[x % self.num_genregs]),
            1: (lambda x:ip_regs[x % self.num_ipregs])
        }

        target_col = prog.prog[const.TARGET]
        source_col = prog.prog[const.SOURCE]
        mode_col = prog.prog[const.MODE]
        op_col = prog.prog[const.OP]

        if not prog.effective_instrs:
            prog.effective_instrs = self.find_introns(target_col, source_col, mode_col)

        for i in prog.effective_instrs:
            source = mode[mode_col[i]](source_col[i])
            do_op(op_col[i], gen_regs, target_col[i], source)

            # for reg in gen_regs:
            #     if (type(reg) != float) and (type(reg) != int) and (type(reg) != np.float64):
            #         pdb.set_trace()

        output = gen_regs[:self.output_dims]
        #self.clear()
        return output

    def clear(self):
        self.gen_regs = [self.default_val]*self.num_genregs

    # Return set w/ indices of effective instructions
    def find_introns(self, target, source, mode):
        marked_instrs = set()
        eff_regs = set(range(self.output_dims))
        instrs = reversed(range(len(target)))
        for i in instrs:
            if target[i] in eff_regs:
                marked_instrs.add(i)
                if mode[i] == 0:
                    eff_regs.add(source[i] % self.num_genregs)
        return sorted(marked_instrs)


def add(target, index, source):
    target[index] += source

def sub(target, index, source):
    target[index] -= source

def mult(target, index, source):
    target[index] *= source

def div(target, index, source):
    target[index] /= (sqrt(1+source**2))

ops = { 0: add, 1: sub, 2: mult, 3: div }


def do_op(op_code, target, index, source):
    try:
        ops[op_code](target, index, source)
    except Exception as e:
        print(e)

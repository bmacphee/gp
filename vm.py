import random, pdb, const
from numpy import sqrt

class Vm:
    def __init__(self, num_genregs, num_ipregs, num_ops, output_dims):
        self.num_genregs = num_genregs
        self.num_ipregs = num_ipregs
        self.num_ops = num_ops
        self.output_dims = output_dims
        self.gen_regs = [0]*num_genregs
        # mode=0: general regs, mode=1: ip regs
        self.mode = {
            0: (lambda x:self.gen_regs[x % self.num_genregs]),
            1: (lambda x:self.ip_regs[x % self.num_ipregs])
        }

    def run_prog(self, prog, ip, remove_introns=True):
        self.prog_len = len(prog[0])
        self.ip_regs = [float(x) for x in ip]
        target_col = prog[const.TARGET]
        source_col = prog[const.SOURCE]
        mode_col = prog[const.MODE]
        op_col = prog[const.OP]
        effective_instrs = self.find_introns(target_col, source_col, mode_col) if remove_introns else range(self.prog_len)

        for i in sorted(effective_instrs):
            #pdb.set_trace()
            target_ind = target_col[i]
            source = self.mode[mode_col[i]](source_col[i])
            self.gen_regs[target_ind] = do_op(op_col[i], self.gen_regs[target_ind], source)

        output = self.gen_regs[:self.output_dims]
        self.clear()
        return output

    def clear(self):
        self.gen_regs = [0]*self.num_genregs

    # Return set w/ indices of effective instructions
    def find_introns(self, target, source, mode):
        marked_instrs = set()
        eff_regs = set(range(self.output_dims))
        instrs = reversed(range(self.prog_len))
        for i in instrs:
            if target[i] in eff_regs:
                marked_instrs.add(i)
                if mode[i] == 0:
                    eff_regs.add(source[i] % self.num_genregs)
        return marked_instrs


def add(target, source):
    return target + source

def sub(target, source):
    return target - source

def mult(target, source):
    return target * source

def div(target, source):
    return target / (sqrt(1+source**2))

ops = { 0: add, 1: sub, 2: mult, 3: div }

def do_op(op_code, target, source):
    try:
        val = ops[op_code](target, source)
    except Exception as e:
        print(e)
        val = target
    return val

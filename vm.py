import random
import pdb
import const


class Vm:
    def __init__(self, num_genregs, num_ipregs, num_ops, output_dims):
        self.num_genregs = num_genregs
        self.num_ipregs = num_ipregs
        self.num_ops = num_ops
        self.output_dims = output_dims
        self.gen_regs = [0]*num_genregs

    def run_prog(self, prog, ip):
        for i in range(0, len(prog[0])):
            target_ind = prog[const.TARGET][i]
            ip_regs = [float(x) for x in ip]
            # mode=0 --> general regs, mode=1 --> ip regs
            if prog[const.MODE][i] == 0:
                source = self.gen_regs[prog[const.SOURCE][i] % self.num_genregs]
            elif prog[const.MODE][i] == 1:
                source = ip_regs[prog[const.SOURCE][i] % self.num_ipregs]
            else:
                raise ValueError('Invalid mode')
            self.gen_regs[target_ind] = do_op(prog[const.OP][i], self.gen_regs[target_ind], source)
        output = self.gen_regs[:self.output_dims]
        self.clear()
        return output

    def clear(self):
        self.gen_regs = [0]*self.num_genregs


def do_op(op_code, target, source):
    if op_code == 0:
        val = (target + source)
    elif op_code == 1:
        val = (target - source)
    elif op_code == 2:
        val = (target * source)
    elif op_code == 3:
        if source != 0:
            val = (target / source)
        else:
            val = target
    else:
        raise ValueError('Invalid op')
    return val

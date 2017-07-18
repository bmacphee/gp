import random, const
import numpy as np
import fitness as fit


def two_prog_recombination(progs):
    progs = [progs[0].copy(), progs[1].copy()]
    assert len(progs) == 2
    prog0, prog1 = progs[0].prog, progs[1].prog
    prog_len = len(prog0[0])
    start_index = random.randint(0, prog_len - 2)
    end_limit = prog_len - 1 if start_index is 0 else prog_len  # avoid swapping whole program
    end_index = random.randint(start_index + 1, end_limit)

    for col in range(len(prog0)):
        prog1[col][start_index:end_index] = prog0[col][start_index:end_index]
        prog0[col][start_index:end_index] = prog1[col][start_index:end_index]
    return [progs[0], progs[1]]

def one_prog_recombination(prog):
    prog_len = len(prog.prog)
    lines = np.random.choice(range(prog_len), 2, replace=False)
    temp = prog.prog[lines[0]]
    prog.prog[lines[0]] = prog.prog[lines[1]]
    prog.prog[lines[1]] = temp
    return prog

#@profile
# Mutation - change 1 value in the program
def mutation(progs, ops, max_vals, effective_mutations=False):
    min_lines, max_lines = 1, len(progs[0].prog[0])

    # One prog input for mutation
    progs = [prog.copy() for prog in progs]
    children = []
    for prog in progs:
        # Test - effective mutations
        if effective_mutations:
            if prog.effective_instrs[0] == -1:
                fit.find_introns(prog)
            num_lines = random.randint(min_lines, min(max_lines, len(prog.effective_instrs)))
            lines = np.random.choice(prog.effective_instrs, size=num_lines, replace=False)

        else:
            num_lines = random.randint(min_lines, max_lines)
            lines = np.random.choice(list(range(max_lines)), size=num_lines, replace=False)
        for index in lines:
            col = random.randint(0, len(prog.prog) - 1)
            orig_val = prog.prog[col][index]
            if col == const.OP:
                options = [x for x in ops if x != orig_val]
            else:
                options = [x for x in range(max_vals[col]) if x != orig_val]
            new_val = np.random.choice(options)
            prog.prog[col][index] = new_val
        children.append(prog)
    return children
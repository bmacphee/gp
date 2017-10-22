import random, const, pdb
import numpy as np
import fitness as fit


class GridLocations:
    def __init__(self, num_sections, train_shape):
        self.grid = {}
        assert np.sqrt(num_sections) == round(np.sqrt(num_sections))
        X = np.array(range(train_shape[0]*train_shape[1])).reshape(train_shape)
        rows = int(np.sqrt(num_sections))
        interval = int(train_shape[0]/rows)
        section = 0
        row_start = 0
        for i in range(rows):
            col_start = 0
            for j in range(rows):
                self.grid[section] = X[row_start:(row_start+interval), col_start:(col_start+interval)]
                col_start += interval
                section += 1
            row_start += interval
        for section in self.grid.keys():
            self.grid[section] = self.grid[section].flatten()


# TODO: does this need to be rows? currently columns (still works)
def two_prog_recombination(progs):
    assert len(progs) == 2
    prog0, prog1 = progs[0].prog, progs[1].prog
    prog_len = len(prog0[0])
    start_index = random.randint(0, prog_len - 2)
    end_limit = prog_len - 1 if start_index is 0 else prog_len  # avoid swapping whole program
    end_index = random.randint(start_index + 1, end_limit)

    for col in range(len(prog0)):
        prog1[col, start_index:end_index] = prog0[col, start_index:end_index]
        prog0[col, start_index:end_index] = prog1[col, start_index:end_index]
    progs[0].clear_effective_instrs()
    progs[1].clear_effective_instrs()
    return [progs[0], progs[1]]

#@profile
def one_prog_recombination(prog):
    prog_len = len(prog.prog[0])
    p = np.asmatrix(prog.prog).T
    lines = random.sample(range(prog_len), 2)

    temp = p[lines[0]].copy()
    p[lines[0]] = p[lines[1]]
    p[lines[1]] = temp
    prog.clear_effective_instrs()
    return prog

# #
# # Mutation - change 1 value in the program
# #@profile
# def mutation(prog, ops, max_vals, grid=None):
#     min_lines, max_lines = 1, len(prog.prog[0])
#
#     # One prog input for mutation
#     children = []
#
#     num_lines = random.randint(min_lines, max_lines)
#     lines = random.sample(range(max_lines), num_lines)
#     l = len(prog.prog)
#     cols = np.random.choice(l, size=num_lines)
#
#     for i in range(num_lines):
#         row, col = lines[i], cols[i]
#         orig_val = prog.prog[col, row]
#         if col == const.OP:
#             options = [x for x in ops if x != orig_val]
#
#         elif grid is not None and col == const.SOURCE and prog.prog[const.MODE, row] == const.IP_MODE_VAL:
#             options = grid.grid[prog.grid_section]
#
#         elif col == const.MODE:
#             options = [1-orig_val]
#         else:
#             options = [x for x in range(max_vals[col]) if x != orig_val]
#
#
#         new_val = random.choice(options)
#         prog.prog[col, row] = new_val
#
#         if grid and (col == const.MODE and new_val == const.IP_MODE_VAL):
#             prog.prog[const.SOURCE, row] = random.choice(grid.grid[prog.grid_section])
#
#     children.append(prog)
#     prog.clear_effective_instrs()
#     return children

#@profile
def mutation(prog, operations, max_vals, grid=None):
    min_lines, max_lines = 1, len(prog.prog[0])

    # One prog input for mutation
    children = []

    num_lines = random.randint(min_lines, max_lines)
    lines = np.random.randint(0, max_lines, size=num_lines)
    l = len(prog.prog)
    cols = np.random.randint(0, l, size=num_lines)

    slots = np.random.random(num_lines)

    for i in range(num_lines):
        row, col = lines[i], cols[i]
        orig_val = prog.prog[col, row]
        if col == const.OP:
            #options = [x for x in ops if x != orig_val]
            options = operations[:]

        elif grid is not None and col == const.SOURCE and prog.prog[const.MODE, row] == const.IP_MODE_VAL:
            options = grid.grid[prog.grid_section]

        elif col == const.MODE:
            options = [1-orig_val]
        #
        else:
            #options = [x for x in range(max_vals[col]) if x != orig_val]
            options = list(range(max_vals[col]))

        # new_val = random.choice(options)

        div = 1/len(options)
        slot = int(slots[i]/div)
        new_val = options[slot]

        if new_val == orig_val:
            new_val = options[slot-1]
        prog.prog[col, row] = new_val

        if grid and (col == const.MODE and new_val == const.IP_MODE_VAL):
            prog.prog[const.SOURCE, row] = random.choice(grid.grid[prog.grid_section])

    children.append(prog)
    prog.clear_effective_instrs()
    return children

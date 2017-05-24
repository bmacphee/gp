import vm, const, pdb, copy
import numpy as np
import gp
from sklearn.metrics import accuracy_score
from importlib import reload
from array import array




class Prog:
    def __init__(self, prog):
        self.prog = prog
        self.effective_instrs = None

def make_target_source(target_ind, source_ind):
    target, source = [target_ind], [source_ind]
    target += [const.GEN_REGS-1]*(const.PROG_LENGTH-1)
    source += [const.GEN_REGS-1]*(const.PROG_LENGTH-1)
    return target, source


def make_vm(num_ipregs=3, num_ops=4, output_dims=3):
    return vm.Vm(const.GEN_REGS, num_ipregs, num_ops, output_dims)

def test_add():
    #v = make_vm()
    op = 0
    ip = np.array([float(x) for x in [10, 20, 30]])
    mode = [[0]*const.PROG_LENGTH,[1]*const.PROG_LENGTH]
    results = [2, 31]
    target, source = make_target_source(1, 2)
    for i in range(len(mode)):
        prog = gp.Prog([target,source,[op]*const.PROG_LENGTH,mode[i]])
        #vm.gen_regs = default_regs[:]
        output = vm.run_prog(prog, ip)
        assert float(output[target[0]]) == float(results[i])
    print('Pass')

def test_sub():
    #v = make_vm()
    op = 1
    ip = np.array([float(x) for x in [10, 20, 30]])
    mode = [[0]*const.PROG_LENGTH,[1]*const.PROG_LENGTH]
    results = [0, -29]
    target, source = make_target_source(1, 2)
    for i in range(len(mode)):
        prog = gp.Prog([target,source,[op]*const.PROG_LENGTH,mode[i]])
        #vm.gen_regs = default_regs[:]
        output = vm.run_prog(prog, ip)
        assert float(output[target[0]]) == float(results[i])
    print('Pass')

def test_mult():
    v = make_vm()
    ip = [10, 20, 30]
    mode = [0,1]
    results = [1, 20]
    target, source = 2, 6
    for i in range(len(mode)):
        prog = gp.Prog([[target],[source],[2],[mode[i]]])
        v.gen_regs = default_regs[:]
        output = v.run_prog(prog, ip)
        assert float(output[target]) == float(results[i])
    print('Pass')

def test_introns():
    #v = make_vm()
    results = [0,2,5]
    prog = gp.Prog([[1, 7, 6, 4, 4, 0],[2, 2, 5, 1, 1, 6],[0,0,0,0,0,0],[1, 1, 0, 0, 1, 0]])
    remain = const.PROG_LENGTH - len(prog.prog[0])
    col = [const.GEN_REGS-1]*remain
    op = [0]*remain
    for i in range(len(prog.prog)):
        if i != const.OP:
            prog.prog[i] += col
        else:
            prog.prog[i] += op

    vm.prog_len = len(prog.prog[0])
    #assert vm.find_introns(prog.prog[const.TARGET], prog.prog[const.SOURCE], prog.prog[const.MODE]) == results
    assert np.array_equal(vm.find_introns(prog.prog[const.TARGET], prog.prog[const.SOURCE], prog.prog[const.MODE]), results)

    assert not prog.effective_instrs
    gp.find_introns(prog)
    assert np.array_equal(prog.effective_instrs, results)
    print('Pass')


def test_div():
    pass

def test_prog_attrs_updated():
    X = [[0, 0, 0, 0, 0]]
    y = [1]
    ip = [10, 20, 30]
    v = make_vm()
    prog = gp.gen_prog(const.PROG_LENGTH, 3)
    assert prog.effective_instrs is None
    assert prog.fitness is None
    target = prog.prog[const.TARGET]
    source = prog.prog[const.SOURCE]
    mode = prog.prog[const.MODE]
    v.run_prog(prog, ip)
    assert prog.effective_instrs is not None
    gp.get_fitness_results([prog], X, y, v, p.accuracy, training=False)
    assert prog.fitness is None
    gp.get_fitness_results([prog], X, y, v, p.accuracy, training=True)
    assert prog.fitness is not None
    print('Pass')

def test_updated_fitness():
    v = make_vm()
    prog = gp.Prog([[0, 0],[0, 0],[0, 0],[0, 0]])
    effective_instrs = [0, 1]
    prog.effective_instrs = effective_instrs
    orig_fitness = 1
    prog.fitness = orig_fitness
    for i in range(2):
        gp.check_fitness_after_variation(prog, v, [i])
        assert prog.effective_instrs == effective_instrs
        assert prog.fitness is None
        prog.fitness = orig_fitness
    prog.prog = [[v.output_dims, v.output_dims],[0, 0],[0, 0],[0, 0]]
    effective_instrs = []
    prog.effective_instrs = effective_instrs
    gp.check_fitness_after_variation(prog, v, [0, 1])
    assert prog.fitness is orig_fitness
    assert prog.effective_instrs == effective_instrs
    print('Pass')

def test_mutation():
    orig = [[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0]]
    parent = gp.Prog(copy.deepcopy(orig))
    for i in range(5):
        child = gp.mutation([parent])[0]
        assert parent.prog == orig

        change_cols = [i for i in range(len(orig)) if child.prog[i] != orig[i]]
        for c in change_cols:
            col = child.prog[c]
            if c != const.OP:
                changed_i = [i for i in range(len(col)) if col[i] != orig[c][i]]
                assert len(changed_i) > 0
                for i in changed_i:
                    orig_val = orig[c][i]
                    assert (col[i] == (orig_val+1)) or (changed_val[0] == (orig_val-1))
    print('Pass')

def test_recombination():
    orig0, orig1 = [[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0]], [[1, 1, 1],[1, 1, 1],[1, 1, 1],[1, 1, 1]]
    prog0, prog1 = gp.Prog(copy.deepcopy(orig0)), gp.Prog(copy.deepcopy(orig1))
    parents = [prog0, prog1]
    children = gp.recombination(parents)
    child0, child1 = children[0], children[1]

    ind = [i for i in range(len(orig0[0])) if orig0[0][i] != child0.prog[0][i]]
    for i in range(1, len(ind)):
        assert ind[i] == ind[i-1]+1

    for i in range(len(orig0)):
        for j in range(len(orig0[0])):
            if j in ind:
                assert child0.prog[i][j] == orig1[i][j]
                assert child1.prog[i][j] == orig0[i][j]
            else:
                assert child0.prog[i][j] == orig0[i][j]
                assert child1.prog[i][j] == orig1[i][j]

    assert prog0.prog == orig0
    assert prog1.prog == orig1
    print('Pass')



def test_results():
    pop_size = 10
    gp.env.pop_size = 10
    gp.env.generations = 100
    #v = make_vm()
    vm.num_genregs = const.GEN_REGS
    vm.num_ipregs = 3
    vm.num_ops = 4
    vm.output_dims = 3

    X_train, y_train = [array('d', [0.0]*3), array('d', [1.0]*3), array('d', [2.0]*3), array('d', [3.0]*3), array('d', [4.0]*3), array('d', [5.0]*3), array('d', [6.0]*3),
                        array('d', [7.0]*3)], [0, 0, 2, 0, 1, 2, 0, 0]
    print(X_train)
    print(y_train)
    selections = [const.Selection.STEADY_STATE_TOURN, const.Selection.BREEDER_MODEL]
    generations = 10
    fitness_eval = gp.accuracy
    gp.env.train_fitness_eval = fitness_eval
    gp.env.test_fitness_eval = fitness_eval

    for selection in selections:
        for i in range(10):
            pop = gp.gen_population(10, 3)
            gp.run_model(X_train, y_train, pop, selection, generations, fitness_eval, show_graph=0)
            results = gp.get_fitness_results(pop, X_train, y_train, fitness_eval, store_fitness='trainset_trainfit')

            for i in range(len(results)):
                prog = pop[i]
                y_pred = gp.predicted_classes(prog, X_train)
                acc = 1-(sum([1 for i in range(len(y_train)) if y_train[i]!=y_pred[i]])/len(y_pred))
                assert round(acc, 5) == round(results[i], 5)
    print('Pass')

def test_fitness_sharing():
    X = [array('d'), array('d'), array('d')]
    y0, y1, y2 = [0, 1, 2], [0, 1, 0], [0, 0, 0]
    prog0, prog1, prog2 = gp.Prog([0]), gp.Prog([0]), gp.Prog([0])
    prog0.train_y_pred, prog1.train_y_pred, prog2.train_y_pred = y0, y1, y2
    y = [0, 1, 2]
    expected = [(1/4)+(1/3)+(1/2), (1/4)+(1/3), (1/4)]
    fitness = gp.fitness_sharing([prog0, prog1, prog2], X, y, store_fitness='trainset_trainfit')
    for i in range(len(fitness)):
        assert fitness[i] == expected[i]
    print('Pass')

def test_predicted_classes():
    pass

def test_avg_detect_rate():
    y = [1, 1, 1, 2, 2, 3]
    y_pred = [1, 1, 1, 2, 0, 0]
    expected = ((3/3) + (1/2) + (0/1)) / 3
    prog = gp.Prog([0])
    rate = gp.avg_detect_rate(prog, y, y_pred)
    assert rate == expected
    print('Pass')

def test_breeder():
    X, y = [array('d')], []
    fitness_eval = lambda:None
    gap = 0.2

    for pop_size in [5, 10, 11, 12]:
        gap_size = int(gap*pop_size)
        pop = gp.gen_population(pop_size, 1)
        fits = list(range(pop_size))
        for i in range(len(pop)):
            pop[i].trainset_trainfit = fits[i]

        bottom_progs = [pop[i] for i in range(gap_size)]
        new_pop = gp.breeder(X, y, pop, fitness_eval, gap=gap)
        assert pop_size == len(new_pop)
        for p in pop:
            if p in bottom_progs:
                assert p not in new_pop
            else:
                assert p in new_pop
    print('Pass')

def test_tourn():
    X, y = [array('d')], []
    fitness_eval = lambda:None
    pop_size = 4

    pop = gp.gen_population(pop_size, 1)
    for p in pop:
        p.effective_instrs = None

    fits = list(range(pop_size))
    for i in range(len(pop)):
        pop[i].trainset_trainfit = fits[i]

    keep_size = pop_size-2
    winners = pop[2:]
    losers = pop[:2]
    new_pop = gp.tournament(X, y, pop, fitness_eval)
    assert pop_size == len(new_pop)

    act_losers = []
    act_winners = []
    for p in pop:
        if not p.effective_instrs:
            act_winners.append(p)
        else:
            act_losers.append(p)
    pdb.set_trace()
    assert len(act_losers) == keep_size
    assert len(act_winners) == keep_size
    for p in winners:
        assert p in act_winners
    for p in losers:
        assert p not in act_winners
    print('Pass')

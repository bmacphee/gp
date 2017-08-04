import const, pdb, copy, pytest
import numpy as np
import cythondir.vm as vm
import data_utils as dutil
import fitness as fit
import gp, config, utils
from sklearn.metrics import accuracy_score
from importlib import reload
from array import array

TEST_DATA = 'data/test/test.data'
TEST_DATA_STRLBL = 'data/test/test_strlbl.data'

EXPECTED_X = [
    [45, 0, 81, 0, -6, 11, 25, 88, 64],
    [46, 0, 96, 0, 52, -4, 40, 44, 4],
    [40, -1, 89, -7, 50, 0, 39, 40, 2],
    [33, 9, 79, 0, 42, -2, 25, 37, 12],
    [38, 3, 109, 0, 72, 7, 1, 36, 36],
    [31, 5, 93, 0, 38, 0, 53, 55, 2],
    [25, 0, 88, 7, 2, 0, 3, 86, 82],
    [25, 0, 78, 1, 26, 0, 23, 52, 30],
    [16, 3, 77, 0, -24, 0, 21, 103, 82],
    [16, 0, 85, 0, 56, 15, 29, 28, 0],
]

EXPECTED_Y = ['4', '4', '4', '3', '3', '3', '2', '2', '1', '1']
EXPECTED_Y_STR = ['test0', 'test3', 'test2', 'test1', 'test0', 'test0', 'test3', 'test2', 'test2', 'test1']


@pytest.fixture
def env():
    env = config.Config()
    setattr(env, 'standardize_method', None)
    setattr(env, 'data_file', TEST_DATA)
    setattr(env, 'test_size', 0.5)
    gp.env = env
    return env


@pytest.fixture
def data(env):
    d = dutil.Data()
    d.load_data(env)
    d.curr_X = d.X_train
    d.curr_y = d.y_train
    d.curr_i = array('i', range(len(d.curr_X)))
    gp.data = d
    return d


@pytest.fixture
def make_data(env):
    data = gp.Data()
    data.load_data(env)
    gp.data = data
    return data


@pytest.fixture
def strlabeldata(data):
    filename = 'data/test/test_strlbl.data'
    return gp.load_data(filename)


@pytest.fixture
def pop():
    progs = [[0], [0], [0], [1]], [[0], [0], [1], [0]], [[0], [0], [2], [0]]
    p = [vm.Prog(i) for i in progs]
    p[0].class_label, p[1].class_label, p[2].class_label = 1, 2, 3
    return p


@pytest.fixture
def hosts():
    return np.array([vm.Host(), vm.Host(), vm.Host(), vm.Host()])



@pytest.fixture
def system(env, pop , hosts):
    return gp.create_system(env, pop, hosts)


def test_load_data():
    d = gp.load_data(TEST_DATA)
    assert len(d) == len(EXPECTED_X)
    for i in range(len(d)):
        line = d[i]
        for j in range(len(line)):
            if j == (len(line) - 1):
                assert line[j] == str(EXPECTED_Y[i])
            else:
                assert line[j] == str(EXPECTED_X[i][j])


def test_get_classes(data, strlabeldata):
    expected_cl = [['test0', 'test1', 'test2', 'test3'], ['1', '2', '3', '4']]
    expected_y = [EXPECTED_Y_STR, EXPECTED_Y]
    d = [strlabeldata, data]

    for i in range(len(d)):
        y = [ex[-1:len(ex)][0] for ex in d[i]]
        assert y == expected_y[i]
        cl = gp.get_classes(y)
        classes = cl.keys()
        assert len(cl) == len(expected_cl[i])
        assert classes == set(expected_cl[i])
        for j, k in enumerate(sorted(classes)):
            assert cl[k] == j


def test_load():
    env = gp.Config()
    env.classes = gp.get_classes(EXPECTED_Y)
    env.standardize_method = lambda x: x


def test_split_data():
    data = []


def test_set_classes(env, data):
    expected_lens = [2, 2, 3, 3]
    expected_cl = util.get_classes(EXPECTED_Y)
    expected_yvals = [expected_cl[label] for label in EXPECTED_Y]
    expected_vals = [
        [EXPECTED_X[9], EXPECTED_X[8]],
        [EXPECTED_X[7], EXPECTED_X[6]],
        [EXPECTED_X[5], EXPECTED_X[4], EXPECTED_X[3]],
        [EXPECTED_X[2], EXPECTED_X[1], EXPECTED_X[0]]
    ]

    # data.classes = gp.get_classes(EXPECTED_Y)

    assert data.data_by_classes == {}
    data.X_train = EXPECTED_X
    data.y_train = EXPECTED_Y
    data.set_classes(EXPECTED_X, expected_yvals)
    data_cl = data.data_by_classes
    assert data_cl is not None
    assert len(data_cl) == len(expected_lens)

    for i, cl in enumerate(sorted(set(expected_yvals))):
        assert len(data_cl[cl]) == expected_lens[i]
        for j in data_cl[cl]:
            ex = data.X_train[j]
            assert ex in expected_vals[i]
            expected_vals[i].remove(ex)
        assert len(expected_vals[i]) == 0


def gen_prog(target, source, op, mod):
    t = [target]
    t += [const.GEN_REGS - 1] * (const.PROG_LENGTH - 1)
    s = [source]
    s += [const.GEN_REGS - 1] * (const.PROG_LENGTH - 1)
    o = [op]
    o += [0] * (const.PROG_LENGTH - 1)
    m = [mod]
    m += [0] * (const.PROG_LENGTH - 1)
    prog = gp.gen_prog([t, s, o, m])
    return prog


@pytest.yield_fixture
def vm_config():
    vm_vals = vm.get_vals()
    yield
    vm.init(vm_vals[0], vm_vals[1], vm_vals[2])


def test_add():
    target, source, op, mode = 0, 0, 0, 1
    p = gen_prog(target, source, op, mode)
    vm.init(const.GEN_REGS, 1, 1)
    ip = np.array([5], dtype=np.float64)
    output = vm.run_prog(p, ip)
    assert output == [6.0]


def test_sub():
    target, source, op, mode = 0, 0, 1, 1
    p = gen_prog(target, source, op, mode)
    vm.init(const.GEN_REGS, 1, 1)
    ip = np.array([5])
    output = vm.run_prog(p, ip)
    assert output == [-4.0]


def test_mult():
    target, source, op, mode = 0, 0, 2, 1
    p = gen_prog(target, source, op, mode)
    vm.init(const.GEN_REGS, 1, 1)
    ip = np.array([5])
    output = vm.run_prog(p, ip)
    assert output == [5.0]


def test_div():
    pass


def test_mean_var():
    X_train = [[0, 0], [1, 10], [2, 20]]
    X_test = [[10, 1], [20, 2], [30, 3]]
    means = [1, 10]
    stds = [0.8164965809, 8.1649658093]
    alpha = 1
    expected_train_vals = [
        [-1.2247, -1.2247],
        [0.0, 0.0],
        [1.2247, 1.2247]
    ]
    expected_test_vals = [
        [11.0227, -1.1023],
        [23.2702, -0.9798],
        [35.5176, -0.8573]
    ]
    standardized = gp.standardize(X_train, X_test, alpha=alpha, method=const.StandardizeMethod.MEAN_VARIANCE)
    stdX_train, stdX_test = np.round(standardized[0], 4).tolist(), np.round(standardized[1], 4).tolist()
    assert np.array_equal(stdX_train, expected_train_vals)
    assert np.array_equal(stdX_test, expected_test_vals)


def test_linear_trans():
    pass


def test_introns():
    results = [0, 2, 5]
    prog = vm.Prog([[1, 7, 6, 4, 4, 0], [2, 2, 5, 1, 1, 6], [0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 0]])
    remain = const.PROG_LENGTH - len(prog.prog[0])
    col = [const.GEN_REGS - 1] * remain
    op = [0] * remain
    for i in range(len(prog.prog)):
        if i != const.OP:
            prog.prog[i] += col
        else:
            prog.prog[i] += op

    vm.prog_len = len(prog.prog[0])
    # assert vm.find_introns(prog.prog[const.TARGET], prog.prog[const.SOURCE], prog.prog[const.MODE]) == results
    assert np.array_equal(vm.find_introns(prog.prog[const.TARGET], prog.prog[const.SOURCE], prog.prog[const.MODE]),
                          results)

    assert not prog.effective_instrs
    gp.find_introns(prog)
    assert np.array_equal(prog.effective_instrs, results)


# Fix these up when data types are decided on (remove converting to lists)
def test_mutation(env):
    vals = [range(const.GEN_REGS), range(env.num_ipregs), env.ops, [0, 1]]
    orig = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # parent = vm.Prog(copy.deepcopy(orig))
    # Need deep copy?
    parent = gp.gen_prog(copy.deepcopy(orig))
    for i in range(10):
        child = gp.mutation([parent])[0]
        parent_prog_as_list = [x.tolist() for x in parent.prog]
        assert parent_prog_as_list == orig

        change_cols = [i for i in range(len(orig)) if list(child.prog[i]) != orig[i]]
        for c in change_cols:
            col = child.prog[c]
            changed_i = [i for i in range(len(col)) if col[i] != orig[c][i]]
            assert len(changed_i) > 0
            for i in changed_i:
                # orig_val = orig[c][i]
                # assert (col[i] == (orig_val+1)) or (col[i]  == (orig_val-1))
                assert (col[i] in vals[c])


def test_recombination():
    orig0, orig1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    # prog0, prog1 = vm.Prog(copy.deepcopy(orig0)), vm.Prog(copy.deepcopy(orig1))
    prog0, prog1 = gp.gen_prog(copy.deepcopy(orig0)), gp.gen_prog(copy.deepcopy(orig1))
    parents = [prog0, prog1]
    children = gp.two_prog_recombination(parents)
    child0, child1 = children[0], children[1]

    ind = [i for i in range(len(orig0[0])) if orig0[0][i] != child0.prog[0][i]]
    for i in range(1, len(ind)):
        assert ind[i] == ind[i - 1] + 1

    for i in range(len(orig0)):
        for j in range(len(orig0[0])):
            if j in ind:
                assert child0.prog[i][j] == orig1[i][j]
                assert child1.prog[i][j] == orig0[i][j]
            else:
                assert child0.prog[i][j] == orig0[i][j]
                assert child1.prog[i][j] == orig1[i][j]

    assert [x.tolist() for x in prog0.prog] == orig0
    assert [x.tolist() for x in prog1.prog] == orig1


def test_results(env):
    pop_size = 10
    gp.env.pop_size = 10
    gp.env.generations = 100
    # v = make_vm()
    vm.num_genregs = const.GEN_REGS
    vm.num_ipregs = 3
    vm.num_ops = 4
    vm.output_dims = 3

    X_train, y_train = [array('d', [0.0] * 3), array('d', [1.0] * 3), array('d', [2.0] * 3), array('d', [3.0] * 3),
                        array('d', [4.0] * 3), array('d', [5.0] * 3), array('d', [6.0] * 3),
                        array('d', [7.0] * 3)], [0, 0, 2, 0, 1, 2, 0, 0]
    print(X_train)
    print(y_train)
    selections = [const.Selection.STEADY_STATE_TOURN, const.Selection.BREEDER_MODEL]
    generations = 10
    fitness_eval = gp.accuracy
    gp.env.train_fitness_eval = fitness_eval
    gp.env.test_fitness_eval = fitness_eval

    for selection in selections:
        for i in range(10):
            pop = gp.gen_population(10)
            gp.run_model(X_train, y_train, pop, selection, generations, fitness_eval, show_graph=0)
            results = gp.get_fitness_results(pop, X_train, y_train, fitness_eval, store_fitness='trainset_trainfit')

            for i in range(len(results)):
                prog = pop[i]
                y_pred = gp.predicted_classes(prog, X_train)
                acc = 1 - (sum([1 for i in range(len(y_train)) if y_train[i] != y_pred[i]]) / len(y_pred))
                assert round(acc, 5) == round(results[i], 5)


def test_fitness_sharing():
    X = [array('d'), array('d'), array('d')]
    y0, y1, y2 = [0, 1, 2], [0, 1, 0], [0, 0, 0]
    prog0, prog1, prog2 = vm.Prog([0]), vm.Prog([0]), vm.Prog([0])
    # prog0.train_y_pred, prog1.train_y_pred, prog2.train_y_pred = y0, y1, y2
    y = [0, 1, 2]
    expected = [(1 / 4) + (1 / 3) + (1 / 2), (1 / 4) + (1 / 3), (1 / 4)]
    fitness = fit.fitness_sharing([prog0, prog1, prog2], X, y)
    for i in range(len(fitness)):
        assert fitness[i] == expected[i]


def test_predicted_classes():
    pass


def test_avg_detect_rate():
    y = array('i', [1, 1, 1, 2, 2, 3])
    y_pred = [1, 1, 1, 2, 0, 0]
    expected = ((3 / 3) + (1 / 2) + (0 / 1)) / 3
    prog = vm.Prog([[0]])
    rate = fit.avg_detect_rate(prog, y, y_pred)
    assert rate == expected


def test_breeder(env, data):
    env.use_subset = 0
    X, y = [array('d')], []
    fitness_eval = lambda: None
    gap = 0.2

    for pop_size in [5, 10, 11, 12]:
        gap_size = int(gap * pop_size)
        pop = gp.gen_population(pop_size)
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


def test_tourn(env, data):
    env.use_subset = 0
    X, y = [array('d')], []
    fitness_eval = lambda: None
    pop_size = 4
    win_size = 2

    pop = gp.gen_population(pop_size)
    for p in pop:
        p.effective_instrs = None

    fits = list(range(pop_size))
    for i in range(len(pop)):
        pop[i].trainset_trainfit = fits[i]

    keep_size = pop_size - win_size
    winners = pop[win_size:]
    losers = pop[:win_size]
    new_pop = gp.tournament(X, y, pop, fitness_eval)
    assert pop_size == len(new_pop)

    act_losers = []
    act_winners = []
    # effective_instrs will be calculated + assigned only for the children
    for p in pop:
        if not p.effective_instrs:
            act_winners.append(p)
        else:
            act_losers.append(p)
    assert len(act_losers) == keep_size
    assert len(act_winners) == win_size
    for p in winners:
        assert p in act_winners
    for p in losers:
        assert p not in act_winners


def test_host_y_pred(env):
    env.bid_gp, env.tangled_graphs, env.point_fitness = 1, 0, 0
    ip = np.array([[45.0, 0.0, 8.0]])
    vm.init(8, 3, 1, 1, 0, 1)

    progs = [[0], [0], [0], [1]], [[0], [0], [1], [0]], [[0], [0], [2], [0]]
    x_inds = array('i', [0])
    pop = []
    for i, p in enumerate(progs):
        prog = vm.Prog(p)
        prog.class_label = i
        pop.append(prog)

    pop = np.array(pop)
    hosts = np.array([vm.Host()])
    hosts[0].set_progs(array('i', [0, 1, 2]))
    host_i = array('i', range(len(hosts)))
    y_pred = vm.host_y_pred(pop, hosts, ip, x_inds, 0, host_i)
    assert y_pred[0] == pop[0].class_label

    hosts[0].set_progs(array('i', [1, 2]))
    y_pred = vm.host_y_pred(pop, hosts, ip, x_inds, 0, host_i)
    assert y_pred[0] == pop[2].class_label


def test_point_fitness(env, data):
    env.bid_gp = 0
    env.use_subset = 1
    env.point_fitness = 1
    data.X_train = [[0], [1], [2], [3], [4], [5]]
    data.y_train = [0, 1, 2, 3, 4, 5]
    y_pred = np.array([[4, 4, 4, 4, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    index = [0, 1, 2, 3, 4]
    pts = [data.X_train[i] for i in index]
    y_act = array('i', [data.y_train[i] for i in index])
    data.curr_X = pts[:]
    data.curr_y = y_act[:]
    data.curr_i = index[:]
    vm.set_curr_ypred(y_pred)
    gp.gen_points(data, env, 1)

    for i in range(4):
        assert data.curr_X[i] == pts[i]
        assert data.curr_y[i] == y_act[i]

    assert data.curr_X[-1] == data.X_train[-1]
    assert data.curr_y[-1] == data.y_train[-1]

    y_pred = np.array([[1, 1, 2, 3, 4], [1, 1, 2, 3, 4], [1, 1, 2, 3, 4], [1, 1, 2, 3, 4], [1, 1, 2, 3, 4]])
    vm.set_curr_ypred(y_pred)
    gp.gen_points(data, env, 1)
    assert data.curr_X[0] == data.X_train[4]
    assert data.curr_y[0] == data.y_train[4]


def test_graph_host_y_pred(env):
    env.bid_gp, env.tangled_graphs, env.point_fitness = 1, 1, 0
    ip = np.array([[45.0, 0.0, 8.0]])
    vm.init(8, 3, 1, 1, 1, 1)

    progs = [[0], [0], [0], [1]], [[0], [0], [1], [0]], [[0], [0], [2], [0]]
    x_inds = array('i', [0])
    pop = []
    for i, p in enumerate(progs):
        prog = vm.Prog(p)
        prog.class_label = i+1
        pop.append(prog)

    pop = np.array(pop)
    hosts = np.array([vm.Host(), vm.Host(), vm.Host()])
    hosts[0].set_progs(array('i', [0, 1, 2]))
    hosts[1].set_progs(array('i', [1]))
    hosts[2].set_progs(array('i', [2]))
    host_i = array('i', range(len(hosts)))

    y_pred = vm.host_y_pred(pop, hosts, ip, x_inds, 0, host_i)
    for i in range(len(hosts)):
        assert y_pred[i] == pop[i].class_label

    pop[0].atomic_action = 0
    host_i = host_i[:1]
    y_pred = vm.host_y_pred(pop, hosts, ip, x_inds, 0, host_i)
    assert y_pred[0] == pop[1].class_label

    pop[1].atomic_action = 0
    y_pred = vm.host_y_pred(pop, hosts, ip, x_inds, 0, host_i)
    assert y_pred[0] == pop[2].class_label

    pop[0].atomic_action = 1
    y_pred = vm.host_y_pred(pop, hosts, ip, x_inds, 0, host_i)
    assert y_pred[0] == pop[0].class_label


def test_root_teams(env, pop, hosts, data):
    pop_len = len(pop)
    env.bid_gp, env.tangled_graphs, env.point_fitness = 1, 1, 0
    system = gp.create_system(gp.env, pop, np.array([hosts[0], None]))
    assert isinstance(system, gp.GraphSystem)
    assert system.root_teams() == [0]
    for p in pop:
        assert p.atomic_action == 1
    for h in hosts:
        h.set_progs(array('i', [0, 1, 2]))

    new_host = hosts[0].copy()
    gp.prob_check = lambda x: True
    gp.copy_change_bid = lambda *x: x[0].copy()
    gp.ops.one_prog_recombination = lambda *x: None
    gp.ops.mutation = lambda *x: None

    gp.modify_symbionts(system, [new_host], data.curr_X, data.curr_i)
    assert len(system.pop) == pop_len*2
    assert system.hosts[0].num_refs == 2
    actions = [p.atomic_action for p in system.pop]
    assert actions.count(1) == 4
    assert actions.count(0) == 2
    assert len(system.root_teams()) == 0
    utils.set_arr([1], system.hosts, new_host)
    assert system.root_teams() == [1]

    unused = [i for i in range(len(system.pop)) if system.pop[i].atomic_action == 0]
    assert unused == [3, 4]
    gp.clear_unused_symbionts(unused, system)
    assert system.hosts[0].num_refs == 0
    assert system.root_teams() == [0, 1]


def test_num_refs():
    pass





def test_is_duplicate(env):
    vm.init(8, 3, 1, 1)  # 8 gen regs, 3 ip regs, 1 out dims, 1 bid, 1 trainsize
    progs = [[0], [0], [0], [1]]  # Target: 0 (gen regs), Source: 0 (ip), Op: +
    ip = np.array([[1.0, 1.0, 1.0]])
    prog = vm.Prog(progs)
    pop = np.asarray(prog)
    hosts = np.asarray(vm.Host())
    hosts[0].set_progs(array('i', [0]))
    x_inds = np.asarray([0])

    vm.host_y_pred(pop, hosts, )



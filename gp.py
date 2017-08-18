import random, sys, time, numpy as np, pdb
import const, config, graph, utils, cythondir.vm as vm, var_ops as ops, fitness as fit, data_utils as dutil, systems
from importlib import reload
from array import array
np.core.arrayprint._line_width=120

TESTING = 0

'''
Generating initial programs
'''


def gen_prog(pr):
    # List of desired program columns
    if type(pr) == list:
        assert len(pr) == 4
        prog = vm.Prog([])
        prog.prog = [
            array('i', pr[0]),
            array('i', pr[1]),
            array('i', pr[2]),
            array('i', pr[3])
        ]
        return prog

    # Generate random program given a prog length
    else:
        assert type(pr) == int
    prog = list(range(const.MODE + 1))
    prog[const.TARGET] = array('i', np.random.randint(const.GEN_REGS, size=pr))
    prog[const.SOURCE] = array('i', np.random.randint(max(const.GEN_REGS, env.num_ipregs), size=pr))
    prog[const.OP] = array('i', np.random.choice(env.ops, size=pr))
    prog[const.MODE] = array('i', np.random.randint(2, size=pr))
    return vm.Prog(prog)


def gen_population(env):
    if env.bid_gp:
        pop_num = (env.pop_size - int(env.pop_size * env.host_gap)) * env.min_teamsize
        total = env.max_teamsize * env.pop_size
    else:
        pop_num = env.pop_size
        total = env.pop_size
    pop = [gen_prog(env.prog_length) for _ in range(0, pop_num)]
    for p in pop:
        vm.set_introns(p)

    pop += [None] * (total - len(pop))
    return np.array(pop)


# @profile
def gen_hosts(pop, data, env):
    if not data.classes:
        raise AttributeError('Data not loaded')

    classes = list(data.classes.values())
    num_hosts = int(env.host_size * env.host_gap)
    hosts = [vm.Host() for _ in range(num_hosts)]

    for i, host in enumerate(hosts):
        host.index_num = i
        pop_index = i * env.min_teamsize
        progs = []
        for j in range(pop_index, pop_index + env.min_teamsize):
            progs.append(j)
        progs = array('i', progs)
        host.set_progs(progs)
        options = classes[:]
        for prog_i in progs:
            prog = pop[prog_i]
            label = random.choice(options)
            prog.action = [prog.atomic_action, label]
            options.remove(label)

    min_size = env.min_teamsize - env.start_teamsize
    max_size = env.max_teamsize - env.start_teamsize
    for host in hosts:
        size = random.randint(min_size, max_size)
        if size > 0:
            options = [x for x in list(range(len(pop))) if x not in host.progs_i and pop[x] is not None]
            host.add_progs(array('i', np.random.choice(options, size, replace=False)))

    hosts += [None] * (env.host_size - num_hosts)
    init_hosts(np.array(hosts), data, pop)
    return np.array(hosts)


def gen_points(data, env, after_first_run=0):
    if env.bid_gp:
        data.last_X_train = data.data_i[:env.num_saved_vals]

    if env.use_subset:
        if env.point_fitness and after_first_run:
            orig_ind = data.data_i[:]

            num_pts = len(data.curr_X)
            partition = num_pts - int(num_pts * env.point_gap)
            point_fit = vm.point_fitness(data.curr_y)
            ranked_index = utils.get_ranked_index(point_fit)
            bottom_i = ranked_index[:-partition]
            max_val = len(data.X_train)

            new_inds = []
            for i in bottom_i:
                ind = np.random.randint(0, max_val)
                while ind in orig_ind:
                    ind = np.random.randint(0, max_val)
                    new_inds.append(ind)

                data.data_i[i] = ind
                data.curr_X[i] = data.X_train[ind]
                data.curr_y[i] = data.y_train[ind]

        else:
            data.curr_X, data.curr_y, data.data_i = env.subset_sampling(data, env.subset_size)
            data.data_i = env.subset_sampling(data, env.subset_size)[2]

    elif not after_first_run:
        data.curr_X, data.curr_y = data.X_train, data.y_train
        data.data_i = array('i', list(range(len(data.X_train))))


def create_system(env):
    pop = gen_population(env)
    hosts = gen_hosts(pop, data, env) if env.bid_gp else None
    if env.bid_gp and env.tangled_graphs:
        return systems.GraphSystem(pop, hosts)
    elif env.bid_gp:
        return systems.BidSystem(pop, hosts)
    return systems.System(pop, hosts)


def init_hosts(hosts, data, pop):
    data.curr_X, data.curr_y, data.data_i = dutil.even_data_subset(data, env.subset_size)
    vm.host_y_pred(pop, hosts, data.curr_X, data.data_i, 0, 1, array('i', np.nonzero(hosts)[0]))


def init_vm(env, data):
    vm.init(env.prog_length, const.GEN_REGS, env.num_ipregs, env.output_dims, env.bid_gp, env.tangled_graphs,
            len(data.X_train), len(data.X_test))


# Steady state tournament for selection
def tournament(data, env, system, fitness_eval):
    pop = system.pop
    X, y = data.curr_X, data.curr_y
    selected_i = np.random.choice(range(len(pop)), const.TOURNAMENT_SIZE, replace=False)
    results = fit.fitness_results([pop[i] for i in selected_i], X, y, fitness_eval, )
    cutoff = int(const.TOURNAMENT_SIZE / 2)

    ranked_results = utils.get_ranked_index(results)
    winners_i = [selected_i[i] for i in ranked_results[cutoff:]]
    losers_i = [selected_i[i] for i in ranked_results[:cutoff]]
    parents = [pop[i] for i in winners_i]

    var_op = np.random.choice([0, 1], p=env.var_op_probs)
    if var_op == 0:
        progs = ops.mutation([parents[0].copy()], env.ops, env.max_vals) + ops.mutation([parents[1].copy()], env.ops,
                                                                                        env.max_vals)
    elif var_op == 1:
        progs = ops.two_prog_recombination([p.copy() for p in parents])

    utils.set_arr(losers_i, pop, progs)


# Breeder model for selection
def breeder(data, env, system, fitness_eval):
    if system.hosts is None:
        prog_breeder(data, env, system, fitness_eval)
    else:
        host_breeder(data, env, system, fitness_eval)


def prog_breeder(data, env, system, fitness_eval):
    results = system.trainset_fitness_results(data.curr_X, data.curr_y, env.train_fitness)
    pop_size = len(system.pop)
    partition = pop_size - int(pop_size * env.breeder_gap)
    ranked_index = utils.get_ranked_index(results)
    bottom_i = ranked_index[:-partition]
    new_progs = []

    while len(new_progs) < len(bottom_i):
        var_op = 0 if len(new_progs) == (pop_size - 1) else np.random.choice([0, 1], p=env.var_op_probs)
        if var_op == 0:
            new_progs += ops.mutation([np.random.choice(system.pop).copy()], env.ops, env.max_vals)
        elif var_op == 1:
            parents = [p.copy() for p in np.random.choice(system.pop, 2, replace=False)]
            new_progs += ops.two_prog_recombination(parents)
    utils.set_arr(bottom_i, system.pop, new_progs)


# @profile
def host_breeder(data, env, system, fitness_eval):
    X, y = data.curr_X, data.curr_y
    last_X = [data.X_train[i] for i in data.last_X_train]

    none_vals = np.where(system.hosts == np.array(None))[0]
    num_none_vals = none_vals.size
    new_hosts = make_hosts(system, num_none_vals)
    modify_symbionts(system, new_hosts, last_X, env.action_change_probs)

    for i in range(len(new_hosts)):
        try:
            new_hosts[i].index_num = none_vals[i]
        except IndexError:
            pdb.set_trace()
    utils.set_arr(none_vals, system.hosts, new_hosts)

    for i in range(len(system.hosts)):
        assert system.hosts[i].index_num == i
        for x in system.hosts[i].progs_i:
            if system.pop[x].atomic_action == 0:
                assert system.hosts[i].index_num != system.pop[x].class_label

    results = system.trainset_fitness_results(X, y, env.train_fitness, data_i=data.data_i)
    size = len(system.root_hosts()) if isinstance(system, systems.GraphSystem) else env.host_size
    partition = int(size - int(size * env.host_gap))
    bottom_i = utils.get_ranked_index(results)[:-partition]
    to_set = [system.root_hosts()[i] for i in bottom_i] if isinstance(system, systems.GraphSystem) else bottom_i
    utils.set_arr(to_set, system.hosts, None)

    clear_inactive_progs(system, env.min_teamsize, env.max_teamsize)
    clear_unused_symbionts(find_unused_symbionts(system), system)


# @profile
# Remove symbionts no longer indexed by any hosts as a consequence of host deletion
def clear_unused_symbionts(unused, system):
    if isinstance(system, systems.GraphSystem):
        for prog in [system.pop[i] for i in unused if system.pop[i].atomic_action == 0]:
            host = system.hosts[prog.class_label]
            if host is not None:
                try:
                    assert host.index_num not in system.root_hosts()
                    prog.action = [1, prog.class_label]  # To decrement host num_refs
                except:
                    pdb.set_trace()
    utils.set_arr(unused, system.pop, None)


# @profile
def clear_inactive_progs(system, min_size, max_size):
    nonzero = np.nonzero(system.pop)[0]
    prog_ids = array('i', [-1] * len(nonzero))
    for i, j in enumerate(nonzero):
        prog_ids[i] = system.pop[j].prog_id

    for host in system.curr_hosts():
        if host.progs_i.size == max_size:
            host.clear_inactive(system.pop, min_size)
        assert len(host.progs_i) > 0


# @profile
def find_unused_symbionts(system):
    all_referenced = set()
    for host in system.curr_hosts():
        all_referenced.update(host.progs_i)
    return [i for i in range(len(system.pop)) if i not in all_referenced and system.pop[i] is not None]


# @profile
def make_hosts(system, num_new):
    new_hosts = [h.copy() for h in np.random.choice(system.curr_hosts(), num_new)]
    for host in new_hosts:
        i = 1
        curr_progs = host.progs_i.base.tolist()

        # Remove symbionts
        while (random.random() <= (env.prob_removal ** (i - 1))) and (len(curr_progs) > env.min_teamsize):
            remove = np.random.choice(curr_progs)
            curr_progs.remove(remove)
            i += 1

        # Add symbionts
        i = 1
        options = [x for x in range(len(system.pop)) if x not in curr_progs and system.pop[x] is not None]
        while (random.random() <= (env.prob_add ** (i - 1))) and (len(curr_progs) < env.max_teamsize):
            if options:
                add = np.random.choice(options)
                curr_progs.append(add)
            else:
                break
        host.set_progs(array('i', curr_progs))
    return new_hosts


# @profile
def modify_symbionts(system, new_hosts, X, action_change_probs):
    unused_i = [i for i in range(len(system.pop)) if system.pop[i] is None]
    for host in new_hosts:
        changed = 0
        progs = host.progs_i.base.tolist()

        while not changed:
            for i in host.progs_i:
                if utils.prob_check(env.prob_modify):
                    symb = system.pop[i]
                    new = copy_change_bid(symb, system, X)

                    # Test to change action
                    atomic_action = 1
                    new_label = symb.class_label
                    if utils.prob_check(env.prob_change_action):
                        # Check if host programs have atomic actions
                        if isinstance(system, systems.GraphSystem):
                            atomic_exists = [system.pop[i].atomic_action for i in progs]
                            if (atomic_exists.count(1) > 1) and utils.prob_check(action_change_probs[0]):
                                atomic_action = 0

                        if (atomic_action == 1) and utils.prob_check(action_change_probs[1]):
                            new_label = np.random.choice(
                                [cl for cl in data.classes.values() if cl != new.class_label])
                        elif atomic_action == 0:
                            # Note: this is only selecting from already present hosts
                            new_label = np.random.choice([h.index_num for h in system.curr_hosts()])
                        new.action = [atomic_action, new_label]


                    new_index = unused_i.pop()
                    system.pop[new_index] = new
                    progs.remove(i)
                    progs.append(new_index)
                    changed = 1
        host.set_progs(array('i', progs))


# @profile
def copy_change_bid(symb, system, X):
    used_pop = np.asarray([system.pop[i] for i in range(len(system.pop)) if system.pop[i] is not None])
    try:
        new = symb.copy()
    except:
        pdb.set_trace()
    duplicate = 1
    new.action = [1, new.class_label]  # Set to 1 for register checking - don't need graphs for this
    temp_hosts = np.asarray([vm.Host()])
    temp_hosts[0].set_progs(array('i', [0]))
    test_pop = np.asarray([new])
    X = np.asarray(X)
    while duplicate:
        ops.mutation([new], env.ops, env.max_vals)
        ops.one_prog_recombination(new)
        vm.host_y_pred(test_pop, temp_hosts, X, None, 0, 1, array('i', [0]))  # Is this right?
        duplicate = new.is_duplicate(used_pop)
    new.action = [symb.atomic_action, new.class_label]
    return new


def run_model(data, system, env):
    start = time.time()
    if env.file_prefix is None:
        env.file_prefix = utils.filenum()
    graph_iter = 0
    stats = dutil.Results()
    stats.init_percentages(data.classes)

    if env.selection == const.Selection.STEADY_STATE_TOURN:
        select = tournament
    elif env.selection == const.Selection.BREEDER_MODEL:
        select = breeder
    else:
        raise ValueError('Invalid selection: {}'.format(env.selection))
    print_info(env)

    for i in range(env.generations):
        if not env.bid_gp:
            assert len(system.pop) == env.pop_size  # Testing
        if (i % 1 == 0):
            print('.', end='')
            sys.stdout.flush()

        # Run a generation of GP
        gen_points(data, env, after_first_run=i)
        X, y, X_ind = data.curr_X, data.curr_y, data.data_i
        select(data, env, system, env.train_fitness)

        # Run train/test fitness evaluations for data to be graphed
        if (i % env.graph_step == 0) or (i == (env.generations - 1)):
            graph_iter += 1

            # Get top training fitness on training data
            if env.to_graph['top_trainfit_in_trainset'] or (env.train_fitness == env.test_fitness):
                trainset_with_trainfit = system.trainset_fitness_results(X, y, env.train_fitness, data_i=X_ind)
                stats.update_trainfit_trainset(trainset_with_trainfit)

            # Get top testing fitness on training
            if (env.train_fitness == env.test_fitness) and not env.use_validation:
                trainset_with_testfit = trainset_with_trainfit
            else:
                if env.use_validation:
                    X, y, X_ind = env.subset_sampling(data, env.validation_size)
                    if i == 0:
                        data.act_valid_size = len(X)
                trainset_with_testfit = system.trainset_fitness_results(X, y, env.test_fitness, data_i=X_ind)

            stats.update_testfit_trainset(trainset_with_testfit)
            # Get testing fitness on testing data, using the host/program with max testing fitness on the training data
            testset_with_testfit = system.testset_fitness_results(data.X_test, data.y_test, env.test_fitness,
                                                                  trainset_with_testfit)
            stats.update_testfit_testset(testset_with_testfit[0], i)

            # Get the percentages of each class correctly identified in the test set
            top = utils.get_ranked_index(trainset_with_testfit)[-1]
            if isinstance(system, systems.BidSystem):
                top = system.root_hosts()[top]
                stats.update_prog_num(get_top_prog(system.curr_hosts(), trainset_with_testfit)[0])
            stats.update_percentages(fit.class_percentages(system, data.X_test, data.y_test, data.classes, 1,
                                                           hosts_i=[top]))

        # Create/save graphs
        if not TESTING:
            if (env.graph_save_step is not None and ((i % env.graph_save_step == 0)) or (i == env.generations - 1)):
                graph.make_graphs(graph_iter, env, data, stats, system.pop, system.hosts, system.root_hosts())
            if (env.json_save_step is not None) and ((i % env.json_save_step == 0) or (i == env.generations - 1)):
                stats.save_objs(system, data, env)
    print_info(env)
    print("Max fitness: {}, generation {}\nTime: {}".format(stats.max_fitness, stats.max_fitness_gen,
                                                            time.time() - start))
    return system, stats


def get_top_prog(pop, results):
    ind = utils.get_ranked_index(results)[-1]
    return pop[ind:ind + 1]


def print_info(env):
    print('Population size: {}\nGenerations: {}\nData: {}\nSelection Replacement: {}\nAlpha: {}\nBid: {}\n'
          'Point Fitness: {}\nGraphs: {}\nFile num: {}\n'.format(env.pop_size, env.generations, env.data_file,
                                                                 env.selection.name,
                                                                 env.alpha, env.bid_gp, env.point_fitness,
                                                                 env.tangled_graphs, env.file_prefix))


# For testing with interpreter - move later
env = config.Config()
data = dutil.Data()
data.load_data(env)
init_vm(env, data)
syst = create_system(env)


def main():
    global syst
    # cProfile.runctx('run_model(data, syst, env)', globals(), locals(), 'Profile.prof')
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()
    syst, stats = run_model(data, syst, env)


if __name__ == '__main__':
    main()

import random, sys, time, numpy as np, pdb, jsonpickle as jp, os
import const, config, graph, utils, cythondir.vm as vm, var_ops, fitness as fit, data_utils as dutil, systems
from importlib import reload
from array import array

# For debugging
np.core.arrayprint._line_width = 120
TESTING = 0

'''
Generating initial programs
'''


def gen_prog(pr, env, data, grid_section=None):
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
    # if data.grid:
    #     grid_sections = list(range(env.grid_sections))
    prog = list(range(const.MODE + 1))
    prog[const.TARGET] = array('i', np.random.randint(const.GEN_REGS, size=pr))
    prog[const.OP] = array('i', np.random.choice(env.ops, size=pr))
    prog[const.MODE] = array('i', np.random.randint(2, size=pr))
    prog[const.SOURCE] = array('i', np.random.randint(max(const.GEN_REGS, data.num_ipregs), size=pr))

    program = vm.Prog(prog)
    if data.grid:
        if grid_section is not None:
            program.grid_section = grid_section
        else:
            program.grid_section = np.random.choice(range(env.grid_sections))
        ip_sources = np.nonzero(program.prog[const.MODE])[0]
        options = data.grid.grid[program.grid_section]
        for i in ip_sources:
            program.prog[const.SOURCE][i] = np.random.choice(options)

    return program


def gen_population(env, data):
    if env.bid_gp:
        pop_num = (env.pop_size - int(env.pop_size * env.host_gap)) * env.min_teamsize
        total = env.max_teamsize * env.pop_size
    else:
        pop_num = env.pop_size
        total = env.pop_size

    if env.grid_sections:
        sections = [i % env.grid_sections for i in range(env.pop_size)]
    else:
        sections = [None] * env.pop_size

    pop = [gen_prog(env.prog_length, env, data, grid_section=sections[i]) for i in range(0, pop_num)]
    for prog in pop:
        vm.set_introns(prog)

    for i in range(len(pop)):
        pop[i].prog_id = i

    pop += [None] * (total - len(pop))
    return np.array(pop)


def gen_hosts(pop, data, env):
    if not data.classes:
        raise AttributeError('Data not loaded')

    classes = list(data.classes.values())
    num_hosts = env.host_size - int(env.host_size * env.host_gap)
    hosts = [vm.Host() for _ in range(num_hosts)]

    for i, host in enumerate(hosts):
        host.index_num = i
        pop_index = i * env.min_teamsize
        progs = []
        for j in range(pop_index, pop_index + env.min_teamsize):
            progs.append(j)
        host.set_progs(array('i', progs))

        first_prog = pop[progs[0]]
        label = classes[i % len(classes)]
        first_prog.action = [first_prog.atomic_action, label]

        options = [cl for cl in classes if cl != label]
        # If using atomic action size limit - should this still only add unique programs labels?
        if env.limit_atomic_actions:
            options = np.random.choice(options, size=env.atomic_per_host - 1, replace=False)
            options = np.insert(options, 0, label).tolist()
            host.atomic_actions_allowed = np.asarray(options, dtype=np.int32)
        else:
            host.atomic_actions_allowed = np.array(classes, dtype=np.int32)

        for prog_i in progs[1:]:
            prog = pop[prog_i]
            label = random.choice(options)
            prog.action = [prog.atomic_action, label]
            options.remove(label)

    min_size = env.min_teamsize - env.start_teamsize
    max_size = env.max_start_teamsize - env.start_teamsize
    for host in hosts:
        size = random.randint(min_size, max_size)
        if size > 0:
            if env.limit_atomic_actions:
                options = [x for x in list(range(len(pop))) if x not in host.progs_i and pop[x] is not None and pop[
                    x].class_label in host.atomic_actions_allowed]
            else:
                options = [x for x in list(range(len(pop))) if x not in host.progs_i and pop[x] is not None]
            host.add_progs(array('i', np.random.choice(options, size, replace=False)))

    init_hosts(np.array(hosts), data, pop)
    hosts += [None] * (env.host_size - num_hosts)
    return np.array(hosts)


def gen_points(data, env):
    data.last_X_train = data.data_i[:env.num_saved_vals]

    if not env.use_subset:
        data.curr_X, data.curr_y = data.X_train, data.y_train
        data.data_i = np.arange(len(data.X_train), dtype=np.int32)
        return

    if not env.point_fitness:
        data.curr_X, data.curr_y, data.data_i = env.subset_sampling(data, env.subset_size)
    else:
        bottom_i = get_worst_pts(data.curr_y, data, env)

        for i in bottom_i:
            options = data.data_by_classes[data.curr_y[i]]
            ind = np.random.choice(options)
            while ind in data.data_i:
                ind = np.random.choice(options)

            data.data_i[i] = ind
            data.curr_X[i] = data.X_train[ind]
            data.curr_y[i] = data.y_train[ind]

    if not data.act_subset_size:
        data.act_subset_size = len(data.curr_X)


def get_worst_pts(y, data, env):
    pt_fitness = vm.point_fitness(y).base
    y = np.asarray(y)
    cl_size = int(env.subset_size / len(data.classes))
    partition = cl_size - int(cl_size * env.point_gap)
    if partition == cl_size:
        partition = 1

    inds = []

    for cl in data.classes:
        class_pts = np.where(y == data.classes[cl])[0]
        ranked_index = utils.get_ranked_index(pt_fitness[class_pts])
        bottom_i = [class_pts[x] for x in ranked_index[:-partition]]
        inds += bottom_i
    return inds


def create_system(env, data):
    if env.grid_sections:
        grid = var_ops.GridLocations(env.grid_sections, env.data_shape)
        data.grid = grid
    pop = gen_population(env, data)

    if env.bid_gp:
        hosts = gen_hosts(pop, data, env)
        if env.tangled_graphs:
            return systems.GraphSystem(pop, hosts)
        return systems.BidSystem(pop, hosts)
    return systems.System(pop)


def init_hosts(hosts, data, pop):
    data.curr_X, data.curr_y, data.data_i = dutil.even_data_subset(data, env.subset_size)
    vm.host_y_pred(pop, hosts, data.curr_X, data.data_i, 0, 1, array('i', range(len(hosts))))


def init_vm(env, data):
    vm.init(const.GEN_REGS, data.output_dims, env.bid_gp, env.tangled_graphs, len(data.X_train), len(data.X_test))


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
        progs = var_ops.mutation([parents[0].copy()], env.ops, env.max_vals) + var_ops.mutation([parents[1].copy()], env.ops,
                                                                                                env.max_vals)
    elif var_op == 1:
        progs = var_ops.two_prog_recombination([p.copy() for p in parents])

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
            new_progs += var_ops.mutation(np.random.choice(system.pop).copy(), env.ops, data.max_vals)
        elif var_op == 1:
            parents = [p.copy() for p in np.random.choice(system.pop, 2, replace=False)]
            new_progs += var_ops.two_prog_recombination(parents)
    utils.set_arr(bottom_i, system.pop, new_progs)


#profile
def host_breeder(data, env, system, fitness_eval):
    X, y = data.curr_X, data.curr_y
    none_vals = np.where(system.hosts == np.array(None))[0]
    num_none_vals = none_vals.size
    new_hosts = make_hosts(system, num_none_vals)
    modify_symbionts(system, new_hosts, env, data)
    # NOTE: If this goes before modify_symbionts, new programs index new hosts
    utils.set_arr(none_vals, system.hosts, new_hosts)
    for i in range(len(new_hosts)):
        new_hosts[i].index_num = none_vals[i]

    results = system.trainset_fitness_results(X, y, env.train_fitness, data_i=data.data_i)
    size = len(system.root_hosts)
    partition = int(size - int(size * env.host_gap))
    bottom_i = utils.get_ranked_index(results, root_hosts_i=system.root_hosts)[:-partition]
    utils.set_arr(bottom_i, system.hosts, [None] * len(bottom_i))

    if not TESTING:
        clear_inactive_progs(system, env.min_teamsize, env.max_teamsize)
    clear_unused_symbionts(system)


# ##profile
# Remove symbionts no longer indexed by any hosts as a consequence of host deletion
def clear_unused_symbionts(system):
    unused = find_unused_symbionts(system)
    if isinstance(system, systems.GraphSystem):
        for prog in [system.pop[i] for i in unused if system.pop[i].atomic_action == 0]:
            host = system.hosts[prog.class_label]
            if host is not None:
                assert host.index_num not in system.root_hosts
                prog.action = [1, prog.class_label]  # To decrement host num_refs

    utils.set_arr(unused, system.pop, [None] * len(unused))


def find_unused_symbionts(system):
    all_referenced = set()
    for host in system.curr_hosts:
        all_referenced.update(host.progs_i)
    return [i for i in range(len(system.pop)) if i not in all_referenced and system.pop[i] is not None]


# ##profile
def clear_inactive_progs(system, min_size, max_size):
    nonzero = np.nonzero(system.pop)[0]
    prog_ids = array('i', [-1] * len(nonzero))
    for i, j in enumerate(nonzero):
        prog_ids[i] = system.pop[j].prog_id

    for host in system.curr_hosts:
        if host.progs_i.size == max_size:
            host.clear_inactive(system.pop, min_size)
        assert len(host.progs_i) >= env.min_teamsize

#profile
def make_hosts(system, num_new):
    options = np.nonzero(syst.pop)[0]
    new_hosts = [h.copy() for h in system.hosts[np.random.choice(system.root_hosts, num_new)]]
    for host in new_hosts:
        i = 1
        curr_progs = host.progs_i.base.tolist()
        host_options = [p for p in options if system.pop[p].class_label in host.atomic_actions_allowed]
        curr_options = np.setdiff1d(host_options, curr_progs).tolist()
        # Remove symbionts
        while (random.random() <= (env.prob_removal ** (i - 1))) and (len(curr_progs) > env.min_teamsize):
            remove = random.choice(curr_progs)
            curr_progs.remove(remove)
            i += 1
        # Add symbionts
        i = 1
        while (random.random() <= (env.prob_add ** (i - 1))) and (len(curr_progs) < env.max_teamsize):
            if curr_options:
                add = random.choice(curr_options)
                curr_progs.append(add)
                curr_options.remove(add)
                i += 1
            else:
                break
        host.set_progs(array('i', curr_progs))
    return new_hosts


#profile
def modify_symbionts(system, new_hosts, env, data):
    test_X = np.array([data.X_train[i] for i in data.last_X_train])
    unused_i = [i for i in range(len(system.pop)) if system.pop[i] is None]
    for host in new_hosts:
        changed = 0
        host_progs = host.progs_i.base.tolist()
        probs = list(env.modify_probs.keys())
        wts = [env.modify_probs[i] for i in probs]

        while not changed:
            for prog_i in host.progs_i:
                if utils.prob_check(env.prob_modify):
                    symb = system.pop[prog_i]
                    to_change = np.random.choice(probs, p=wts)

                    if to_change == 'action_change':
                        # new = symb.copy()
                        # if isinstance(system, systems.GraphSystem) and atomic_action == 1:
                        #     # Check if host programs have atomic actions
                        #     num_atomic = [system.pop[i].atomic_action for i in host_progs].count(1)
                        #     if utils.prob_check(env.prob_atomic_change) and (num_atomic > 1):
                        #         atomic_action = 0
                        # if atomic_action == 1:
                        #     new_label = np.random.choice(
                        #         [cl for cl in data.classes.values() if cl != new.class_label])
                        # elif atomic_action == 0:
                        #     # new_label = np.random.choice([h.index_num for h in system.curr_hosts])
                        #     new_label = np.random.choice(np.nonzero(system.hosts)[0])
                        # new.action = [atomic_action, new_label]
                        new = copy_change_action(host, symb, system, env)

                    elif to_change == 'modify':
                        new = copy_change_bid(symb, system, env, data, test_X)

                    elif to_change == 'grid_change':
                        new = copy_change_grid_section(symb, data.grid)

                    if TESTING:
                        new.grid_section = symb.prog_id

                    new_index = unused_i.pop()
                    system.pop[new_index] = new
                    new.prog_id = new_index
                    host_progs[host_progs.index(prog_i)] = new_index
                    changed = 1
        host.set_progs(array('i', host_progs))

#profile
def copy_change_action(host, symb, system, env):
    atomic_action = symb.atomic_action
    new = symb.copy()
    host_progs = host.progs_i.base.tolist()

    if isinstance(system, systems.GraphSystem) and atomic_action == 1:
        # Check if host programs have atomic actions
        num_atomic = [system.pop[i].atomic_action for i in host_progs].count(1)
        if (num_atomic > 1) and utils.prob_check(env.prob_atomic_change):
            atomic_action = 0

    if atomic_action == 1:
        atomic_options = [cl for cl in host.atomic_actions_allowed if cl != new.class_label]
        new_label = np.random.choice(atomic_options)
    elif atomic_action == 0:
        new_label = np.random.choice(np.nonzero(system.hosts)[0])

    new.action = [atomic_action, new_label]
    return new

#profile
def copy_change_grid_section(symb, grid):
    try:
        new = symb.copy()
        orig_section = symb.grid_section
        new.grid_section = np.random.choice([x for x in grid.grid.keys() if x != orig_section])

        for j, mode_val in enumerate(new.prog[const.MODE]):
            if mode_val == const.IP_MODE_VAL:
                ind = np.where(grid.grid[orig_section] == new.prog[const.SOURCE][j])[0]
                new.prog[const.SOURCE][j] = grid.grid[new.grid_section][ind]
    except:
        pdb.set_trace()
    return new


#profile
def copy_change_bid(symb, system, env, data, X):
    used_pop = system.pop[np.nonzero(system.pop)[0]]
    new = symb.copy()
    duplicate = 1
    new.action = [1, new.class_label]  # Set to 1 for register checking - don't need graphs for this

    while duplicate:
        var_ops.mutation(new, env.ops, data.max_vals, grid=data.grid)
        var_ops.one_prog_recombination(new)
        if env.check_bid_diff:
            duplicate = new.is_duplicate(used_pop, X)
        else:
            duplicate = 0
    new.action = [symb.atomic_action, new.class_label]
    return new


def update_stats(env, data, system, stats, gen):
    # Get top training fitness on training data
    stats.generation = gen

    X, y, X_ind = data.curr_X, data.curr_y, data.data_i
    if env.to_graph['top_trainfit_in_trainset'] or (env.train_fitness == env.test_fitness):
        trainset_with_trainfit = system.trainset_fitness_results(X, y, env.train_fitness, data_i=X_ind)
        stats.update_trainfit_trainset(trainset_with_trainfit)

    # Get top testing fitness on training
    if (env.train_fitness == env.test_fitness) and not env.use_validation:
        trainset_with_testfit = trainset_with_trainfit
    else:
        if env.use_validation:
            # X, y, X_ind = env.subset_sampling(data, env.validation_size)
            X = data.validation_X
            y = data.validation_y
            X_ind = array('i', range(len(X)))
            data.act_valid_size = len(X)
        trainset_with_testfit = system.trainset_fitness_results(X, y, env.test_fitness, data_i=X_ind,
                                                                validation=env.use_validation)
    stats.update_testfit_trainset(trainset_with_testfit, system)
    top = stats.top_i

    # Get testing fitness on testing data, using the host/program with max testing fitness on the training data
    testset_with_testfit = system.testset_fitness_results(data.X_test, data.y_test, env.test_fitness, top)
    stats.update_testfit_testset(testset_with_testfit[0], gen)

    if isinstance(system, systems.BidSystem):
        nodes = None
        if isinstance(system, systems.GraphSystem):
            nodes, edges, path = dutil.get_top_host_graph(system, env, data, stats)
            systems.print_top_host(nodes, edges, path, data, env.file_prefix)
        stats.update_prog_num(system.hosts[top], nodes=nodes)

    # Get the percentages of each class correctly identified in the test set
    stats.update_percentages(fit.class_percentages(system, data.X_test, data.y_test, data.classes, 1,
                                                   hosts_i=[top]))

    # Eventually move this??
    stats.last_y_pred = \
        system.y_pred(data.X_test, 1, array('i', [stats.top_i]), array('i', range(len(data.X_test)))).base[0]


def run_model(data, system, env, stats):
    start = time.time()
    if env.file_prefix is None:
        env.file_prefix = utils.filenum()

    if env.selection == const.Selection.BREEDER_MODEL:
        select = breeder
    elif env.selection == const.Selection.STEADY_STATE_TOURN:
        select = tournament

    print_info(env)
    start_gen = env.curr_generation

    try:
        for i in range(start_gen, env.generations):
            env.curr_generation += 1

            if (i % 10 == 0):
                print_val = '.'
                if (i % 100 == 0):
                    print_val = i
                print('{}'.format(print_val), end='')
                sys.stdout.flush()

            # Run a generation of GP
            gen_points(data, env)
            select(data, env, system, env.train_fitness)

            # Run train/test fitness evaluations for data to be graphed
            if (i % env.graph_step == 0) or (i == (env.generations - 1)):
                update_stats(env, data, system, stats, i)

            if not TESTING:
                if utils.should_save(i, env.graph_save_step, env.generations):
                    graph.make_graphs(env, data, stats, system)
                if utils.should_save(i, env.json_save_step, env.generations):
                    stats.save_objs(system, data, env)

                # Save env/consts for testing reference
                # (consts should technically be moved to Config, but ideally won't be variable eventually)
                if i == 0:
                    consts = {}
                    store = [('BID_DIFF', const.BID_DIFF), ('GEN_REGS', const.GEN_REGS),
                             ('PT_FITNESS', const.PT_FITNESS)]
                    for n in store:
                        consts[n[0]] = n[1]

                    filepath = utils.make_filename(const.IMAGE_DIR, env.file_prefix, 'env')
                    with open(filepath, 'w') as f:
                        f.write(jp.encode(env))
                        f.write(jp.encode(consts))


    except (KeyboardInterrupt):
        save = input('\nSave? y/n\n')
        if save == 'y':
            update_stats(env, data, system, stats, i)
            graph.make_graphs(env, data, stats, system)
            stats.save_objs(system, data, env)
    finally:
        print_info(env)
        print("Max fitness: {}, generation {}\nTime: {}".format(stats.max_fitness, stats.max_fitness_gen,
                                                                time.time() - start))
    return system, stats


def get_top_prog(pop, results):
    ind = utils.get_ranked_index(results)[-1]
    return pop[ind:ind + 1]


def print_info(env):
    print('Population size: {}\nGenerations: {}\nData: {}\nSelection Replacement: {}\nAlpha: {}\nBid: {}\n'
          'Point Fitness: {}\nGraphs: {}\nTrain Fitness: {}\nFile num: {}\n'.format(env.pop_size, env.generations,
                                                                                    env.data_file,
                                                                                    env.selection.name,
                                                                                    env.alpha, env.bid_gp,
                                                                                    env.point_fitness,
                                                                                    env.tangled_graphs,
                                                                                    env.train_fitness,
                                                                                    env.file_prefix))


# For testing with interpreter - move later
env = config.Config()
if env.random_seed is None:
    env.random_seed = np.random.randint(0, 2000000)
np.random.seed(env.random_seed)
random.seed(env.random_seed)

data = dutil.Data()
data.load_data(env)
init_vm(env, data)
syst = create_system(env, data)
stats = dutil.Results()
stats.init_percentages(data.classes)


def main():
    global syst, stats, data, env
    syst, stats = run_model(data, syst, env, stats)


if __name__ == '__main__':
    main()

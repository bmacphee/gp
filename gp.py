import const, random, sys, time, utils, matplotlib, os, config, numpy as np, pdb
import cythondir.vm as vm, var_ops as ops, fitness as fit, data_utils as dutil

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from importlib import reload
from array import array

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


def gen_population(pop_num):
    pop = [gen_prog(const.PROG_LENGTH) for _ in range(0, pop_num)]
    for p in pop:
        vm.set_introns(p)
    return pop


# @profile
def gen_hosts(pop, data):
    if not data.classes:
        raise AttributeError('Data not loaded')

    classes = list(data.classes.values())
    num_hosts = int(env.host_size * env.host_gap)
    hosts = [vm.Host() for _ in range(num_hosts)]

    for i, host in enumerate(hosts):
        pop_index = i * 2
        progs = array('i', [pop_index, pop_index + 1])
        host.set_progs(progs)
        options = classes[:]
        for prog_i in progs:
            prog = pop[prog_i]
            prog.class_label = np.random.choice(options)
            options.remove(prog.class_label)

    min_size = env.min_teamsize - env.start_teamsize
    max_size = env.max_teamsize - env.start_teamsize
    for host in hosts:
        size = random.randint(min_size, max_size)
        if size > 0:
            options = [x for x in list(range(len(pop))) if x not in host.progs_i]
            host.add_progs(array('i', np.random.choice(options, size, replace=False)))

    init_hosts(np.array(hosts), data, pop)
    hosts += [None] * (env.host_size - num_hosts)

    return np.array(hosts)


def gen_points(data, env, after_first_run=0):
    if env.bid_gp:
        data.last_X_train = data.curr_i[:const.NUM_SAVED_VALS]

    if env.use_subset:
        if env.point_fitness and after_first_run:
            orig_ind = data.curr_i[:]

            num_pts = len(data.curr_X)
            partition = num_pts - int(num_pts * env.point_gap)
            point_fit = vm.point_fitness(data.curr_y)
            ranked_index = get_ranked_index(point_fit)
            bottom_i = ranked_index[:-partition]
            max_val = len(data.X_train)

            for i in bottom_i:
                ind = np.random.randint(0, max_val)
                while ind in orig_ind:
                    ind = np.random.randint(0, max_val)
                data.curr_i[i] = ind
                data.curr_X[i] = data.X_train[ind]
                data.curr_y[i] = data.y_train[ind]
        else:
            data.curr_X, data.curr_y, data.curr_i = env.subset_sampling(data, env.subset_size)
            data.curr_i = env.subset_sampling(data, env.subset_size)[2]

    elif not after_first_run:
        data.curr_X, data.curr_y = data.X_train, data.y_train
        data.curr_i = array('i', list(range(len(data.X_train))))


def init_hosts(hosts, data, pop):
    data.curr_X, data.curr_y, data.curr_i = dutil.even_data_subset(data, env.subset_size)
    vm.host_y_pred(np.asarray(pop), hosts, data.curr_X, data.curr_i, 1)


'''
Results
'''


# @profile
def fitness_results(pop, X, y, fitness_eval, hosts=None, curr_i=None):
    pop_arr = np.asarray(pop)
    if fitness_eval.__name__ == 'fitness_sharing':
        results = fit.fitness_sharing(pop_arr, X, y, hosts, curr_i)
    else:
        all_y_pred = vm.y_pred(pop_arr, X) if hosts is None else vm.host_y_pred(pop_arr, hosts, X, None, 0)
        results = [fitness_eval(pop[i], y, all_y_pred[i]) for i in range(len(all_y_pred))]
    return results


'''
Selection
'''


# @profile
# Steady state tournament for selection
def tournament(data, env, pop, fitness_eval, hosts=None):
    X, y = data.curr_X, data.curr_y
    selected_i = np.random.choice(range(len(pop)), const.TOURNAMENT_SIZE, replace=False)
    results = fitness_results([pop[i] for i in selected_i], X, y, fitness_eval)
    cutoff = int(const.TOURNAMENT_SIZE / 2)

    ranked_results = get_ranked_index(results)
    winners_i = [selected_i[i] for i in ranked_results[cutoff:]]
    losers_i = [selected_i[i] for i in ranked_results[:cutoff]]
    parents = [pop[i] for i in winners_i]

    var_op = np.random.choice([0, 1], p=env.var_op_probs)
    if var_op == 0:
        progs = ops.mutation([parents[0].copy()], env.ops, env.max_vals) + ops.mutation([parents[1].copy()], env.ops,
                                                                                        env.max_vals)
    elif var_op == 1:
        progs = ops.two_prog_recombination([p.copy() for p in parents])

    set_arr(losers_i, pop, progs)


# Breeder model for selection
# @profile
def breeder(data, env, pop, fitness_eval, hosts=None):
    if hosts is None:
        prog_breeder(data, env, pop, fitness_eval)
    else:
        host_breeder(data, env, pop, fitness_eval, hosts)


def prog_breeder(data, env, pop, fitness_eval):
    results = fitness_results(pop, data.curr_X, data.curr_y, fitness_eval, curr_i=data.curr_i)
    pop_size = len(pop)
    partition = pop_size - int(pop_size * env.breeder_gap)
    ranked_index = get_ranked_index(results)
    bottom_i = ranked_index[:-partition]
    new_progs = []

    while len(new_progs) < len(bottom_i):
        var_op = 0 if len(new_progs) == (pop_size - 1) else np.random.choice([0, 1], p=env.var_op_probs)
        if var_op == 0:
            new_progs += ops.mutation([np.random.choice(pop).copy()], env.ops, env.max_vals)
        elif var_op == 1:
            parents = [p.copy() for p in np.random.choice(pop, 2, replace=False)]
            new_progs += ops.two_prog_recombination(parents)
    set_arr(bottom_i, pop, new_progs)


# @profile
def host_breeder(data, env, pop, fitness_eval, hosts):
    X, y = data.curr_X, data.curr_y
    last_X = [data.X_train[i] for i in data.last_X_train]
    partition = int(env.host_size - int(env.host_size * env.host_gap))
    curr_hosts = get_nonzero(hosts)

    new_hosts = make_hosts(pop, curr_hosts, (env.host_size - partition), last_X, data.last_X_train)
    set_arr(np.nonzero(hosts == np.array(None))[0], hosts, new_hosts)
    results = fitness_results(pop, X, y, fitness_eval, hosts=hosts, curr_i=data.curr_i)
    bottom_i = get_ranked_index(results)[:-partition]
    set_arr(bottom_i, hosts, None)

    curr_hosts = get_nonzero(hosts)
    clear_inactive_progs(pop, curr_hosts, env.max_teamsize)
    # Remove symbionts no longer indexed by any hosts as a consequence of host deletion
    set_arr(find_unused_symbionts(pop, curr_hosts), pop, None)

    while pop[-1] is None:
        del pop[-1]


def clear_inactive_progs(pop, hosts, max_size):
    for i, host in enumerate(hosts):
        host_size = host.progs_i.size
        if host_size == max_size:
            progs_i = host.progs_i
            inactive_progs = [progs_i[i] for i in range(host_size) if pop[progs_i[i]].active == 0]
            if inactive_progs:
                new_progs_i = array('i', [prog for prog in progs_i if prog not in inactive_progs])
                host.set_progs(new_progs_i)


# @profile
def find_unused_symbionts(pop, hosts):
    all_referenced = set()
    for host in hosts:
        all_referenced.update(host.progs_i)
    return [i for i in range(len(pop)) if i not in all_referenced and pop[i] is not None]


# @profile
def make_hosts(pop, hosts, num_new, X, X_i):
    new_hosts = [h.copy() for h in np.random.choice(hosts, num_new)]
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
        options = [x for x in range(len(pop)) if x not in curr_progs and pop[x] is not None]
        while (random.random() <= (env.prob_add ** (i - 1))) and (len(curr_progs) < env.max_teamsize):
            add = np.random.choice(options)
            curr_progs.append(add)
            i += 1
        host.set_progs(array('i', curr_progs))
    modify_symbionts(pop, new_hosts, X, X_i)

    return new_hosts


# @profile
def modify_symbionts(pop, hosts, X, X_i):
    unused_i = [i for i in range(len(pop)) if pop[i] is None]
    for host in hosts:
        changed = 0
        progs = host.progs_i.base.tolist()
        while not changed:
            for i in host.progs_i:
                if random.random() <= env.prob_modify:
                    symb = pop[i]
                    new = copy_change_bid(symb, pop, X, X_i)

                    # Test to change action
                    if random.random() <= env.prob_change_label:
                        new.class_label = np.random.choice(
                            [cl for cl in data.classes.values() if cl != new.class_label])

                    if unused_i:
                        new_index = unused_i.pop()
                        pop[new_index] = new
                    else:
                        new_index = len(pop)
                        pop.append(new)

                    progs.remove(i)
                    progs.append(new_index)
                    changed = 1
        host.set_progs(array('i', progs))


# @profile
def copy_change_bid(symb, pop, X, x_inds):
    used_pop = np.asarray([pop[i] for i in range(len(pop)) if pop[i] is not None])
    new = symb.copy()
    duplicate = 1
    temp_hosts = np.asarray([vm.Host()])
    temp_hosts[0].set_progs(array('i', [0]))
    test_pop = np.asarray([new])
    X = np.asarray(X)

    while duplicate:
        ops.mutation([new], env.ops, env.max_vals)
        ops.one_prog_recombination(new)
        vm.host_y_pred(test_pop, temp_hosts, X, None, 1)
        duplicate = new.is_duplicate(used_pop)

    return new


def run_model(data, pop, env, hosts=None):
    start = time.time()
    filename_prefix = utils.filenum()
    graph_iter = 0
    sample_pop = hosts if env.bid_gp else pop
    stats = dutil.Results()
    stats.init_percentages(data.classes)

    if env.selection == const.Selection.STEADY_STATE_TOURN:
        select = tournament
    elif env.selection == const.Selection.BREEDER_MODEL:
        select = breeder
    else:
        raise ValueError('Invalid selection: {}'.format(env.selection))
    print_info(env, filename_prefix)

    for i in range(env.generations):
        if not env.bid_gp:
            assert len(pop) == env.pop_size  # Testing
        if (i % 10 == 0):
            print('.', end='')
            sys.stdout.flush()

        # Run a generation of GP
        gen_points(data, env, after_first_run=i)
        X, y, X_ind = data.curr_X, data.curr_y, data.curr_i
        select(data, env, pop, env.train_fitness, hosts=hosts)

        # Run train/test fitness evaluations for data to be graphed
        if (i % env.graph_step == 0) or (i == (env.generations - 1)):
            graph_iter += 1
            curr_hosts = get_nonzero(hosts) if hosts is not None else None

            # Get top training fitness on training data
            if env.to_graph['top_trainfit_in_trainset'] or (env.train_fitness == env.test_fitness):
                trainset_with_trainfit = fitness_results(pop, X, y, env.train_fitness, hosts=curr_hosts,
                                                         curr_i=X_ind)
                stats.update_trainfit_trainset(trainset_with_trainfit)

            # Get top testing fitness on training data
            if (env.train_fitness != env.test_fitness):
                if env.use_validation:
                    xvals, yvals, _ = env.subset_sampling(data, env.validation_size)
                    if i == 0:
                        data.act_valid_size = len(xvals)
                else:
                    xvals, yvals = X, y
                trainset_with_testfit = fitness_results(pop, xvals, yvals, env.test_fitness, hosts=curr_hosts)
                stats.update_testfit_trainset(trainset_with_testfit)
            else:
                trainset_with_testfit = trainset_with_trainfit

            # Get testing fitness on testing data, using the host/program with max testing fitness on the training data
            p = pop if env.bid_gp else get_top_prog(sample_pop, trainset_with_testfit)
            h = get_top_prog(curr_hosts, trainset_with_testfit) if env.bid_gp else None
            testset_with_testfit = fitness_results(p, data.X_test, data.y_test, env.test_fitness, hosts=h)
            stats.update_testfit_testset(testset_with_testfit[0], i)

            # Get the percentages of each class correctly identified in the test set
            p = pop if env.bid_gp else p[0]
            h = h[0] if env.bid_gp else None
            stats.update_percentages(fit.class_percentages(p, data.X_test, data.y_test, data.classes, host=h))
            stats.update_prog_num(h)

        # Save the graph
        if not TESTING:
            if ((env.graph_save_step is not None) and (i % env.graph_save_step == 0)) or (i == env.generations - 1):
                graph_params = stats.get_graph_params(env.to_graph)
                graph(filename_prefix, env.graph_step, graph_iter, *graph_params)
                if env.to_graph['percentages']:
                    graph_percs(filename_prefix, env.graph_step, graph_iter, stats.percentages)
                if env.to_graph['cumulative_detect_rate']:
                    cumulative = cumulative_detect_rate(data, pop, hosts, testset_with_testfit)
                    graph_cumulative(filename_prefix, env.graph_step, graph_iter, cumulative)
                plt.close('all')

    print("Max fitness: {}, generation {}\nTime: {}".format(stats.max_fitness, stats.max_fitness_gen,
                                                            time.time() - start))
    stats.save_objs(pop, hosts, data, env, filename_prefix)
    return pop, hosts, stats


def cumulative_detect_rate(data, pop, hosts, testset_with_testfit):
    # Move this later - is calculated twice
    if hosts is not None:
        curr_hosts = get_nonzero(hosts)
        y_pred = vm.host_y_pred(np.asarray(pop), curr_hosts, data.X_test, None, 0)
    else:
        y_pred = vm.y_pred(np.asarray(pop), data.X_test)
    detect_rates = []
    ranked = get_ranked_index(testset_with_testfit)
    top = y_pred[ranked[-1]]

    while len(ranked) > 0:
        addition = y_pred[ranked.pop()]
        top[addition == data.y_test] = addition[addition == data.y_test]
        detect_rate = vm.avg_detect_rate(None, array('i', top), data.y_test)
        detect_rates.append(detect_rate)

        if detect_rate == 1:
            break
    detect_rates += [1.0]*(len(curr_hosts)-len(detect_rates))
    return detect_rates


def graph_cumulative(filename_prefix, graph_step, graph_iter, cumulative):
    pass


def graph(n, graph_step, last_x, top_train_fit_on_train=None, train_means=None, test_means=None,
          top_train_prog_on_test=None, top_test_fit_on_train=None):
    gens = [i * graph_step for i in range(last_x)]
    fig = plt.figure(figsize=(13, 13), dpi=80)
    ax = fig.add_subplot(111)
    valid_train = 'Validation' if env.use_validation else 'Train'
    labels = [
        'Max Train Fitness in Train Set', 'Mean Train Fitness in Train Set', 'Mean Test Fitness in Train Set',
        'Best {} Prog on Test Set (using test_fit)'.format(valid_train),
        'Best Train Prog on {} Set (using test_fit)'.format(valid_train)
    ]

    if top_train_fit_on_train:
        ax.plot(gens, top_train_fit_on_train, label=labels[0])
    if train_means:
        ax.plot(gens, train_means, label=labels[1])
    if test_means:
        ax.plot(gens, test_means, label=labels[2])
    if top_test_fit_on_train:
        ax.plot(gens, top_test_fit_on_train, label=labels[4])
    if top_train_prog_on_test:
        ax.plot(gens, top_train_prog_on_test, label=labels[3])

    subset_str = ', Subset Size: {}'.format(data.act_subset_size) if env.use_subset else ''
    valid_str = ''
    if env.use_validation:
        valid_str = ', Validation Size: {}'.format(data.act_valid_size)
    op_str = ', '.join([const.OPS[x] for x in env.ops])
    plt.title(
        'Data: {}\nSelection: {}, Bid GP: {}, Point Fitness: {}\nPop Size: {}, Generations: {}, Step Size: {}{}{}\n'
        'Training Fitness: {}, Test Fitness: {}\nOps: [{}], Alpha: {}'.format(env.data_file, env.selection.value,
                                                                              env.bid_gp,
                                                                              env.point_fitness,
                                                                              env.pop_size, env.generations,
                                                                              env.graph_step, subset_str, valid_str,
                                                                              env.train_fitness.__name__,
                                                                              env.test_fitness.__name__, op_str,
                                                                              env.alpha))
    if gens[-1] != 0:
        ax.set_xlim(xmax=gens[-1])
    if TESTING == 0:
        filename = '{}_fitness.png'.format(n)
        save_figure(filename, fig)


def graph_percs(n, graph_step, last_x, percentages):
    generations = [i * graph_step for i in range(last_x)]
    fig = plt.figure(figsize=(13, 13), dpi=80)
    ax = fig.add_subplot(111)
    labels = sorted([perc for perc in percentages])
    for l in labels:
        ax.plot(generations, percentages[l], label=l)
    plt.title('% Classes Correct (Training data: {})'.format(env.data_file))
    if generations[-1] != 0:
        ax.set_xlim(xmax=generations[-1])
    if TESTING == 0:
        filename = '{}_classes.png'.format(n)
        save_figure(filename, fig)


def save_figure(filename, fig):
    ax = fig.axes[0]
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    plt.grid(which='both', axis='both')
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymax=1.02)
    ax.set_ylim(ymin=0)
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.tick_params(which='both', width=1)

    date = time.strftime("%d_%m_%Y")
    filepath = os.path.join(const.IMAGE_DIR, date, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.set_size_inches(22, 11)
    fig.savefig(filepath, dpi=100)


def get_ranked_index(results):
    return [x[0] for x in sorted(enumerate(results), key=lambda i: i[1])]


def get_top_prog(pop, results):
    ind = get_ranked_index(results)[-1]
    return pop[ind:ind + 1]


def set_arr(index, arr, vals):
    for i in range(len(index)):
        try:
            arr[index[i]] = vals[i]
        except (TypeError, IndexError):
            arr[index[i]] = vals


def get_nonzero(arr):
    return arr[np.nonzero(arr)[0]]


def print_info(env, filename_prefix):
    print('Population size: {}\nGenerations: {}\nData: {}\nSelection Replacement: {}\nAlpha: {}\nBid: {}\n'
          'Point Fitness: {}\nFile num: {}\n'.format(env.pop_size, env.generations, env.data_file, env.selection.name,
                                                     env.alpha, env.bid_gp, env.point_fitness, filename_prefix))


def init_vm(env, data):
    vm.init(const.GEN_REGS, env.num_ipregs, env.output_dims, env.bid_gp, len(data.X_train))


# For testing with interpreter - move later
env = config.Config()
data = dutil.Data()
if env.data_file:
    data.load_data(env)
    init_vm(env, data)

pop = gen_population(env.pop_size)
hs = gen_hosts(pop, data)


# @profile
def main():
    pop = gen_population(env.pop_size)
    if not env.bid_gp:
        hs = None
    else:
        hs = gen_hosts(pop, data)

    run_model(data, pop, env, hosts=hs)


if __name__ == '__main__':
    main()

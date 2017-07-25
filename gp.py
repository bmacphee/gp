import const, random, sys, time, utils, matplotlib, os, pdb
import numpy as np
import cythondir.vm as vm
import var_ops as ops

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitness as fit
import data_utils as dutil
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from importlib import reload
from array import array

TESTING = 0
data, env = None, None


class Config:
    def __init__(self):
        self.ops = array('i', [0, 1, 2, 3, 4, 5, 6, 7])  # Ops [0:+, 1:-, 2:*, 3:/, 4:sin, 5:e, 6:ln, 7:conditional]
        self.pop_size = 100
        self.generations = 100
        self.graph_step = 10
        self.graph_save_step = 10
        self.data_files = ['data/iris.data', 'data/tic-tac-toe.data', 'data/ann-train.data', 'data/shuttle.trn',
                           'data/MNIST', 'data/gisette_train.data']
        self.data_file = self.data_files[2]
        self.standardize_method = const.StandardizeMethod.MEAN_VARIANCE
        self.selection = const.Selection.BREEDER_MODEL
        self.breeder_gap = 0.2
        self.alpha = 1
        self.use_subset = 1
        self.subset_size = 200
        self.use_validation = self.use_subset
        self.validation_size = self.subset_size
        self.test_size = 0.2
        self.num_ipregs = None
        self.output_dims = None
        self.train_fitness_eval = fit.fitness_sharing
        self.test_fitness_eval = fit.avg_detect_rate
        self.subset_sampling = dutil.even_data_subset
        self.max_vals = []

        self.bid_gp = 1
        self.point_fitness = 0
        self.host_size = self.pop_size
        self.point_gap = 0.2
        self.host_gap = 0.5
        self.prob_removal = 0.7
        self.prob_add = 0.7
        self.prob_modify = 0.2
        self.prob_change_label = 0.1
        self.prob_mutate = 1
        self.prob_swap = 1
        self.max_teamsize = 10
        self.min_teamsize = 2
        self.start_teamsize = 2

        if not self.use_subset:
            self.point_fitness = 0


class Data:
    def __init__(self):
        self.classes = None  # Available classes
        self.data_by_classes = {}  # Data organized by classes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.last_X_train = None
        self.curr_X = None
        self.curr_y = None
        self.curr_i = None

        self.act_subset_size = None
        self.act_valid_size = None

    def load_data(self, config):
        if config.data_file is None:
            raise AttributeError('No data file to load')

        if config.data_file == 'data/MNIST':
            self.X_train, self.y_train, self.X_test, self.y_test = dutil.load_mnist()  # Requires gzip method
            self.classes = dutil.get_classes(self.y_train)

        else:
            data = dutil.load_data(config.data_file)
            try:
                labels = dutil.load_data(const.LABEL_FILES[config.data_file])
                y = [int(i) for sublist in labels for i in sublist]
                X = dutil.preprocess([ex for ex in data])
            except KeyError:
                y = [ex[-1:len(ex)][0] for ex in data]
                X = dutil.preprocess([ex[:len(ex) - 1] for ex in data])
            self.classes = dutil.get_classes(y)
            y = np.array([self.classes[label] for label in y], dtype=np.int32)


            # Data with corresponding test file - load train/test
            try:
                test_data_file =  const.TEST_DATA_FILES[config.data_file]
                test_data = dutil.load_data(test_data_file)
                X_train = X
                self.y_train = y

                try:
                    X_test = dutil.preprocess([ex for ex in test_data])
                    labels = dutil.load_data(const.LABEL_FILES[test_data_file])
                    self.y_test = [self.classes[int(i)] for sublist in labels for i in sublist]
                except KeyError:
                    X_test = dutil.preprocess([ex[:len(ex) - 1] for ex in test_data])
                    self.y_test = np.array([self.classes[label] for label in [ex[-1:len(ex)][0] for ex in test_data]],
                                            dtype=np.int32)
            # Data with no corresponding test file - split into train/test
            except KeyError:
                X_train, X_test, self.y_train, self.y_test = dutil.split_data(X, y, config.test_size)

            if config.standardize_method is not None:
                X_train, X_test = dutil.standardize(X_train, X_test, env.standardize_method)
            self.X_train, self.X_test = np.array(X_train, dtype=np.float64), np.array(X_test, dtype=np.float64)

        config.num_ipregs = len(self.X_train[0])
        config.max_vals = array('i', [const.GEN_REGS, max(const.GEN_REGS, config.num_ipregs), -1, 2])

        if config.bid_gp:
            config.output_dims = 1  # One bid register
        else:
            config.output_dims = len(self.classes)  # Register for each class

    def set_classes(self, X, y):
        for cl in self.classes.values():
            # self.data_by_classes[cl] = [X[i] for i in range(len(X)) if y[i] == cl]
            self.data_by_classes[cl] = [i for i in range(len(X)) if y[i] == cl]


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


#@profile
def gen_hosts(pop, data):
    if not data.classes:
        raise AttributeError('Data not loaded')

    classes = list(data.classes.values())
    num_hosts = int(env.host_size * env.host_gap)
    hosts = [vm.Host() for _ in range(num_hosts)]

    for i, host in enumerate(hosts):
        pop_index = i * 2
        progs = array('i', [pop_index, pop_index + 1])
        # host.progs_i = progs
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

    # hosts += make_hosts(pop, hosts, int(env.host_size - int(env.host_size*env.host_gap)))
    init_hosts(np.array(hosts), data, pop)
    hosts += [None] * (env.host_size - num_hosts)

    return np.array(hosts)


def init_hosts(hosts, data, pop):
    X, y, x_ind = dutil.even_data_subset(data, env.subset_size)
    vm.host_y_pred(np.asarray(pop), hosts, X, np.asarray(x_ind), 1)
    data.curr_X = X
    data.curr_y = y
    data.curr_i = np.array(x_ind) # for initial run, to set last_X_train value


'''
Results
'''


#@profile
def get_fitness_results(pop, X, y, fitness_eval, hosts=None, curr_i=None):
    if fitness_eval.__name__ == 'fitness_sharing':
        results = fit.fitness_sharing(np.asarray(pop), X, y, hosts, np.asarray(curr_i))
    else:
        if hosts is not None:
            all_y_pred = vm.host_y_pred(np.asarray(pop), np.asarray(hosts), X, None, 0)
        else:
            all_y_pred = vm.y_pred(np.asarray(pop), X)
        results = [fitness_eval(pop[i], y, all_y_pred[i]) for i in range(len(all_y_pred))]
    return results


'''
Selection
'''


#@profile
# Steady state tournament for selection
def tournament(data, env, pop, fitness_eval, var_op_probs=[0.5, 0.5], hosts=None):
    X, y = data.curr_X, data.curr_y
    indivs = set()
    while len(indivs) < const.TOURNAMENT_SIZE:
        indivs.add(random.randint(0, len(pop) - 1))
    selected_i = list(indivs)
    results = get_fitness_results([pop[i] for i in selected_i], X, y, fitness_eval)

    ranked_results = get_ranked_index(results)
    winners_i = [selected_i[i] for i in ranked_results[2:]]
    losers_i = [selected_i[i] for i in ranked_results[:2]]
    parents = [pop[i] for i in winners_i]
    for i in sorted(losers_i, reverse=True):
        del pop[i]

    var_op = np.random.choice([0, 1], p=var_op_probs)
    if var_op == 0:
        progs = ops.mutation([parents[0].copy()], env.ops, env.max_vals) + ops.mutation([parents[1].copy()], env.ops, env.max_vals)
    elif var_op == 1:
        progs = ops.two_prog_recombination([p.copy() for p in parents])
    pop += progs

    return pop, hosts


# Breeder model for selection
#@profile
def breeder(data, env, pop, fitness_eval, hosts=None, var_op_probs=[0.5, 0.5]):
    if hosts is None:
        pop = prog_breeder(data, env, pop, fitness_eval, var_op_probs)
    else:
        pop, hosts = host_breeder(data, env, pop, fitness_eval, hosts)

    return pop, hosts


def prog_breeder(data, env, pop, fitness_eval, var_op_probs):
    gap = env.breeder_gap
    results = get_fitness_results(pop, data.curr_X, data.curr_y, fitness_eval, curr_i=data.curr_i)
    partition = len(pop) - int(len(pop) * gap)
    pop_size = len(pop)
    ranked_index = get_ranked_index(results)
    bottom_i = ranked_index[:-partition]
    bottom_i.sort(reverse=True)
    for i in bottom_i:
        del pop[i]

    while len(pop) < pop_size:
        if len(pop) == (pop_size - 1):
            var_op = 0
        else:
            var_op = np.random.choice([0, 1], p=var_op_probs)

        if var_op == 0:
            progs = ops.mutation([np.random.choice(pop).copy()], env.ops, env.max_vals)
        elif var_op == 1:
            parents = [p.copy() for p in np.random.choice(pop, 2, replace=False)]
            progs = ops.two_prog_recombination(parents)
            # progs = ops.two_prog_recombination(np.random.choice(pop, 2, replace=False).tolist())
        pop += progs
    return pop


#@profile
def host_breeder(data, env, pop, fitness_eval, hosts):
    X, y = data.curr_X, data.curr_y
    last_X = [data.X_train[i] for i in data.last_X_train[:50]]

    partition = int(env.host_size - int(env.host_size * env.host_gap))
    curr_hosts = hosts[hosts != np.array(None)]


    new_hosts, pop = make_hosts(pop, curr_hosts, (env.host_size - partition), last_X, data.last_X_train[:50])

    for i, host_i in enumerate([x for x in range(len(hosts)) if x not in np.nonzero(hosts)[0]]):
        hosts[host_i] = new_hosts[i]

    results = get_fitness_results(pop, X, y, fitness_eval, hosts=hosts, curr_i=data.curr_i)
    ranked_index = get_ranked_index(results)
    bottom_i = ranked_index[:-partition]


    for i in bottom_i:
        hosts[i] = None

    clear_inactive_progs(pop, hosts, env.max_teamsize)
    # Remove symbionts no longer indexed by any hosts as a consequence of host deletion
    unused_i = find_unused_symbionts(pop, hosts)
    for i in unused_i:
        pop[i] = None

    while pop[-1] is None:
        del pop[-1]

    return pop, hosts


def clear_inactive_progs(pop, hosts, max_size):
    for i in np.nonzero(hosts)[0]:
        host = hosts[i]
        host_size = host.progs_i.size
        if host_size == max_size:
            progs_i = host.progs_i.base[:]
            inactive_progs = [progs_i[i] for i in range(host_size) if pop[progs_i[i]].active == 0]
            if inactive_progs:
                new_progs_i = array('i', [prog for prog in progs_i if prog not in inactive_progs])
                host.set_progs(new_progs_i)


#@profile
def find_unused_symbionts(pop, hosts):
    all_referenced = set([v for s in [host.progs_i for host in hosts if host is not None] for v in s])
    return [i for i in range(len(pop)) if i not in all_referenced]


#@profile
def make_hosts(pop, hosts, num_new, X, X_i):
    #pdb.set_trace()
    new_hosts = [h.copy() for h in np.random.choice(hosts, num_new)]

    for host in new_hosts:
        i = 1
        test = 1
        curr_progs = host.progs_i.base.tolist()

        # Remove symbionts
        while (test <= (env.prob_removal ** (i - 1))) and (len(curr_progs) > env.min_teamsize):
            remove = np.random.choice(curr_progs)
            curr_progs.remove(remove)
            i += 1
            test = np.random.rand()

        # Add symbionts
        i = 1
        test = 1
        options = [x for x in range(len(pop)) if x not in curr_progs and pop[x] is not None]
        while (test <= (env.prob_add ** (i - 1))) and (len(curr_progs) < env.max_teamsize):
            add = np.random.choice(options)
            curr_progs.append(add)
            i += 1
            test = np.random.rand()
        host.set_progs(array('i', curr_progs))
    modify_symbionts(pop, new_hosts, X, X_i)

    return new_hosts, pop


#@profile
def modify_symbionts(pop, hosts, X, X_i):
    #pdb.set_trace()
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

    return pop


#@profile
def copy_change_bid(symb, pop, X, x_inds):
    #pdb.set_trace()
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

    # ops.mutation([new], env.ops, env.max_vals)
    # ops.one_prog_recombination(new)
    return new


#@profile
def gen_points(data, env, after_first_run=0):
    if env.bid_gp:
        data.last_X_train = data.curr_i[:]

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

    elif not after_first_run:
        data.curr_X, data.curr_y = data.X_train, data.y_train
        data.curr_i = list(range(len(data.X_train)))


def run_model(data, pop, env, hosts=None):
    start = time.time()
    filename_prefix = utils.filenum()
    validation = env.use_validation
    max_fitness_gen, max_fitness, graph_iter = 0, 0, 0
    sample_pop = hosts if env.bid_gp else pop
    X, y, X_test, y_test = data.X_train, data.y_train, data.X_test, data.y_test
    print('File num: {}'.format(filename_prefix))

    # Components to graph
    to_graph = {
        'top_trainfit_in_trainset': 0,  # Top training fitness value in training set
        'train_means': None,  # Mean of training fitness values in training set
        'test_means': 1,  # Mean of testing fitness values in training set
        'top_train_prog_on_test': 1,  # Testing fitness on test set of top training fitness prog in training set
        'top_test_fit_on_train': 1,  # Testing fitness on train set of top training fitness prog in training set
        'percentages': 1
    }
    to_graph['train_means'] = to_graph['top_trainfit_in_trainset']

    if to_graph['percentages']:
        percentages = {}
        for cl in data.classes:
            percentages[cl] = []

    trainset_results_with_trainfit, trainset_results_with_testfit, testset_results_with_testfit = [], [], []
    trainfit_trainset_means, testfit_trainset_means, top_trainfit_in_trainset, top_prog_testfit_on_testset, top_testfit_in_trainset = [], [], [], [], []
    print_info()

    if env.selection == const.Selection.STEADY_STATE_TOURN:
        select = tournament
    elif env.selection == const.Selection.BREEDER_MODEL:
        select = breeder
    else:
        raise ValueError('Invalid selection: {}'.format(env.selection))

    for i in range(env.generations):
        if not env.bid_gp:
            assert len(pop) == env.pop_size  # Testing
        if (i % 100 == 0):
            print('.', end='')
            sys.stdout.flush()

        gen_points(data, env, after_first_run=i)
        X, y, X_ind = data.curr_X, data.curr_y, data.curr_i

        # TODO: remove train examples from this? (use prev. valid data?)
        if env.use_validation:
            X_valid, y_valid, _ = env.subset_sampling(data, env.validation_size)
            if i == 0:
                data.act_valid_size = len(X_valid)

        # Run a generation of GP
        pop, hosts = select(data, env, pop, env.train_fitness_eval, hosts=hosts)

        # Run train/test fitness evaluations for data to be graphed
        if (i % env.graph_step == 0) or (i == (env.generations - 1)):
            graph_iter += 1
            curr_hosts = hosts[hosts != np.array(None)] if hosts is not None else None
            # Get top training fitness on training data
            if to_graph['top_trainfit_in_trainset'] or (env.train_fitness_eval == env.test_fitness_eval):
                trainset_results_with_trainfit = get_fitness_results(pop, X, y, env.train_fitness_eval,
                                                                     hosts=curr_hosts, curr_i=X_ind)
                top_trainfit_in_trainset.append(max(trainset_results_with_trainfit))
                trainfit_trainset_means.append(np.mean(trainset_results_with_trainfit))
            # Get top testing fitness on training data
            if (env.train_fitness_eval != env.test_fitness_eval) and (to_graph['top_test_fit_on_train']
                                                                      or to_graph['test_means']):
                if validation:
                    xvals, yvals = X_valid, y_valid
                else:
                    xvals, yvals = X, y
                trainset_results_with_testfit = get_fitness_results(pop, xvals, yvals, env.test_fitness_eval,
                                                                    hosts=curr_hosts)
                top_testfit_in_trainset.append(max(trainset_results_with_testfit))
                testfit_trainset_means.append(np.mean(trainset_results_with_testfit))
            # Get testing fitness on testing data, using the host/program with max testing fitness on the training data
            if to_graph['top_train_prog_on_test']:
                if env.train_fitness_eval == env.test_fitness_eval:
                    trainset_results_with_testfit = trainset_results_with_trainfit

                ### Fix this
                if env.bid_gp:
                    testset_results_with_testfit = get_fitness_results(pop, X_test, y_test, env.test_fitness_eval,
                                                                       hosts=[get_top_prog(
                                                                           curr_hosts, trainset_results_with_testfit)]
                                                                       )[0]
                else:
                    testset_results_with_testfit = \
                        get_fitness_results([get_top_prog(sample_pop, trainset_results_with_testfit)],
                                            X_test, y_test, env.test_fitness_eval)[0]
                top_prog_testfit_on_testset.append(testset_results_with_testfit)
                if testset_results_with_testfit > max_fitness:
                    max_fitness = testset_results_with_testfit
                    max_fitness_gen = i
            # Get the percentages of each class correctly identified in the test set
            if to_graph['percentages']:
                if env.bid_gp:
                    cp = fit.class_percentages(pop, X_test, y_test, data.classes,
                                               host=get_top_prog(curr_hosts, trainset_results_with_testfit))
                else:
                    cp = fit.class_percentages(get_top_prog(sample_pop, trainset_results_with_testfit), X_test, y_test,
                                               data.classes)
                for p in cp:
                    percentages[p].append(cp[p])

        # Save the graph
        if not TESTING and (((env.graph_save_step is not None) and (i % env.graph_save_step == 0))
                            or (i == env.generations - 1)):
            # Get graph parameters for graphing function - set to None to not display a graph component
            graphparam = [top_trainfit_in_trainset, trainfit_trainset_means, testfit_trainset_means,
                          top_prog_testfit_on_testset, top_testfit_in_trainset]
            graph_inc = [to_graph['top_trainfit_in_trainset'], to_graph['train_means'], to_graph['test_means'],
                         to_graph['top_train_prog_on_test'], to_graph['top_test_fit_on_train']]
            graphparam = list(map(lambda x: graphparam[x] if graph_inc[x] else None, range(len(graphparam))))
            graph(filename_prefix, env.graph_step, graph_iter, graphparam[0], graphparam[1], graphparam[2],
                  graphparam[3], graphparam[4])
            if to_graph['percentages']:
                graph_percs(filename_prefix, env.graph_step, graph_iter, percentages)
            plt.close('all')

    print("Max fitness: {} at generation {}".format(max_fitness, max_fitness_gen))
    print("\nTime: {}".format(time.time() - start))
    return pop, trainset_results_with_trainfit, trainset_results_with_testfit, testset_results_with_testfit


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
                                                                              env.train_fitness_eval.__name__,
                                                                              env.test_fitness_eval.__name__, op_str,
                                                                              env.alpha))
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    plt.grid(which='both', axis='both')
    ax.set_xlim(xmin=0)
    if gens[-1] != 0:
        ax.set_xlim(xmax=gens[-1])
    ax.set_ylim(ymax=1.02)
    ax.set_ylim(ymin=0)
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.tick_params(which='both', width=1)

    if TESTING == 0:
        filename = '{}_fitness.png'.format(n)
        save_figure(filename, fig)


def graph_percs(n, graph_step, last_x, percentages):
    generations = [i * graph_step for i in range(last_x)]
    fig = plt.figure(figsize=(13,13), dpi=80)
    ax = fig.add_subplot(111)
    labels = sorted([perc for perc in percentages])
    for l in labels:
        ax.plot(generations, percentages[l], label=l)
    plt.title('% Classes Correct (Training data: {})'.format(env.data_file))
    plt.legend(bbox_to_anchor=(1.1, 1), fontsize=8)
    plt.grid(which='both', axis='both')
    ax.set_xlim(xmin=0)
    if generations[-1] != 0:
        ax.set_xlim(xmax=generations[-1])
    ax.set_ylim(ymax=1.02)
    ax.set_ylim(ymin=0)
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.tick_params(which='both', width=1)

    if TESTING == 0:
        filename = '{}_classes.png'.format(n)
        save_figure(filename, fig)


def save_figure(filename, fig):
    date = time.strftime("%d_%m_%Y")
    filepath = os.path.join(const.IMAGE_DIR, date, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fig.set_size_inches(22, 11)
    fig.savefig(filepath,dpi=100)


def get_ranked_index(results):
    return [x[0] for x in sorted(enumerate(results), key=lambda i: i[1])]


def get_top_prog(pop, results):
    return pop[get_ranked_index(results)[-1]]


def print_info():
    print('Population size: {}\nGenerations: {}\nData: {}\nSelection Replacement: {}\nAlpha: {}\nBid: {}\n'
          'Point Fitness: {}\n'.format(env.pop_size, env.generations, env.data_file,
                                       env.selection.name, env.alpha, env.bid_gp,
                                       env.point_fitness))


def init_vm(env, data):
    vm.init(const.GEN_REGS, env.num_ipregs, env.output_dims, env.bid_gp, len(data.X_train))


# For testing with interpreter - move later
env = Config()
data = Data()
if env.data_file:
    data.load_data(env)
    init_vm(env, data)

pop = gen_population(env.pop_size)
hs = gen_hosts(pop, data)


#@profile
def main():
    pop = gen_population(env.pop_size)
    if not env.bid_gp:
        hs = None
    else:
        hs = gen_hosts(pop, data)

    run_model(data, pop, env, hosts=hs)


if __name__ == '__main__':
    main()

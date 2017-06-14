import const, random, sys, time, shelve, matplotlib, pdb
import numpy as np
import cythondir.vm as vm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitness as fit
import data_utils as util
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from importlib import reload
from array import array


TESTING = 0
data, env = None, None


class Config:
    def __init__(self):
        self.ops = [0, 1, 2, 3]
        self.pop_size = 100
        self.generations = 100
        self.graph_step = 1
        self.graph_save_step = None
        self.data_files = ['data/iris.data', 'data/tic-tac-toe.data', 'data/ann-train.data', 'data/shuttle.trn',
                           'data/MNIST/train-images.idx3-ubyte']
        self.data_file = self.data_files[1]

        self.standardize_method = const.StandardizeMethod.MEAN_VARIANCE
        self.selection = const.Selection.BREEDER_MODEL
        self.alpha = 1
        self.use_subset = 1
        self.subset_size = 200
        self.use_validation = 1
        self.validation_size = 200
        self.test_size = 0.2
        self.train_fitness_eval = None
        self.test_fitness_eval = None
        self.num_ipregs = None
        self.train_fitness_eval = fit.fitness_sharing
        self.test_fitness_eval = fit.avg_detect_rate
        self.subset_sampling = util.even_data_subset


class Data:
    def __init__(self):
        self.classes = None
        self.data_by_classes = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, config):
        if config.data_file is None:
            return

        data = util.load_data(config.data_file)
        y = [ex[-1:len(ex)][0] for ex in data]
        self.classes = util.get_classes(y)
        y = np.array([self.classes[label] for label in y], dtype=np.int32)
        X = util.preprocess([ex[:len(ex) - 1] for ex in data])

        try:
            test_data = util.load_data(const.TEST_DATA_FILES[config.data_file])
            X_train = X
            X_test = util.preprocess([ex[:len(ex) - 1] for ex in test_data])
            self.y_train = y
            self.y_test = np.array([self.classes[label] for label in [ex[-1:len(ex)][0] for ex in test_data]],
                                   dtype=np.int32)
        except KeyError:
            X_train, X_test, self.y_train, self.y_test = util.split_data(X, y, config.test_size)

        if config.standardize_method is not None:
            X_train, X_test = util.standardize(X_train, X_test, env.standardize_method)
            self.X_train, self.X_test = np.array(X_train, dtype=np.float64), np.array(X_test, dtype=np.float64)
        config.num_ipregs = len(self.X_train[0])
        config.output_dims = len(self.classes)
        config.max_vals = [const.GEN_REGS, max(const.GEN_REGS, config.num_ipregs), max(config.ops), 2]

    def set_classes(self, X, y):
        for cl in self.classes.values():
            self.data_by_classes[cl] = [X[i] for i in range(len(X)) if y[i] == cl]


# class Prog:
#     def __init__(self, prog):
#         self.prog = prog
#         self.effective_instrs = None
#         # Fitness for training eval on train, testing eval on train, testing eval on test
#         self.trainset_trainfit = None
#         self.trainset_testfit = None
#         self.testset_testfit = None
#         self.train_y_pred = None


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
    pop = [gen_prog(const.PROG_LENGTH) for n in range(0, pop_num)]
    for p in pop:
        p.effective_instrs = fit.find_introns(p)
    return pop


'''
Results
'''


#@profile
def get_fitness_results(pop, X, y, fitness_eval, store_fitness=None):
    # results = [fitness_eval(prog, y, predicted_classes(prog, X, v)) if (prog.fitness is None or not training) else prog.fitness for prog in pop]
    if fitness_eval.__name__ == 'fitness_sharing':
        results = fit.fitness_sharing(pop, X, y)
    else:
        # all_y_pred = vm.y_pred(np.array(pop), np.array(X), 0).tolist()
        all_y_pred = vm.y_pred(np.asarray(pop), X)
        # results = [fitness_eval(prog, y, predicted_classes(prog, X), store_fitness=store_fitness)
        #            if ((store_fitness is None) or getattr(prog, store_fitness) == -1)
        #            else getattr(prog, store_fitness) for prog in pop]

        results = [fitness_eval(pop[i], y, all_y_pred[i]) for i in range(len(pop))]
    return results


'''
Variation operators
'''


#@profile
# Recombination - swap sections of 2 programs
def recombination(progs):
    # progs = copy.deepcopy(progs)
    progs = [progs[0].copy(), progs[1].copy()]
    assert len(progs) == 2
    prog0, prog1 = progs[0].prog, progs[1].prog
    prog_len = len(prog0[0])
    start_index = random.randint(0, prog_len - 2)
    end_limit = prog_len - 1 if start_index is 0 else prog_len  # avoid swapping whole program
    end_index = random.randint(start_index + 1, end_limit)

    for col in range(len(prog0)):
        prog1[col][start_index:end_index], prog0[col][start_index:end_index] = prog0[col][start_index:end_index], prog1[
                                                                                                                      col][
                                                                                                                  start_index:end_index]
    # for prog in progs:
    #     check_fitness_after_variation(prog, list(range(start_index, end_index)))
    return [progs[0], progs[1]]


#@profile
# Mutation - change 1 value in the program
def mutation(progs, effective_mutations=False):
    step_size = 1
    min_lines, max_lines = 1, len(progs[0].prog[0])

    # One prog input for mutation
    progs = [progs[0].copy()]
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
            # new_val = orig_val
            if col == const.OP:
                options = [x for x in env.ops if x != orig_val]
                # new_val = np.random.choice(options)
            #
            # elif (col == const.TARGET) and effective_mutations:
            #     options = set([prog.prog[const.TARGET][i] for i in prog.effective_instrs]).difference(set([orig_val]))
            #     new_val = np.random.choice(options)

            else:
                # while new_val == orig_val:
                #     if new_val == 0:
                #         op = 0
                #     elif (new_val == env.max_vals[col]):
                #         op = 1
                #     else:
                #         op = random.randint(0, 1)
                # new_val = (orig_val+step_size) if op is 0 else (orig_val-step_size)
                # Use all vals or use 1-step vals?
                # pdb.set_trace()
                options = [x for x in range(env.max_vals[col]) if x != orig_val]
            new_val = np.random.choice(options)
            prog.prog[col][index] = new_val
        # check_fitness_after_variation(prog, lines)
        children.append(prog)
    return children


'''
Selection
'''


#@profile
# Steady state tournament for selection
def tournament(X, y, pop, fitness_eval, var_op_probs=[0.5, 0.5]):
    indivs = set()
    while len(indivs) < const.TOURNAMENT_SIZE:
        indivs.add(random.randint(0, len(pop) - 1))
    selected_i = list(indivs)
    results = get_fitness_results([pop[i] for i in selected_i], X, y, fitness_eval=fitness_eval,
                                  store_fitness='trainset_trainfit')

    winners_i = []
    while len(winners_i) < 2:
        max_i = results.index(max(results))
        winners_i.append(selected_i[max_i])
        results[max_i] = -float('inf')
    losers_i = [i for i in selected_i if i not in winners_i]
    parents = [pop[i] for i in winners_i]
    for i in sorted(losers_i, reverse=True):
        del pop[i]

    var_op = np.random.choice([0, 1], p=var_op_probs)
    if var_op == 0:
        progs = mutation([parents[0]]) + mutation([parents[1]])
    elif var_op == 1:
        progs = recombination(parents)

    pop += progs
    # if fitness_eval.__name__ == 'fitness_sharing':
    #     for prog in pop[-2:]:
    #         if getattr(prog, 'train_y_pred')[0] == -1:
    #             clear_attrs(pop)
    #             break

    return pop


# Breeder model for selection
#@profile
def breeder(X, y, pop, fitness_eval, gap=0.2, var_op_probs=[0.5, 0.5]):
    env.pop_size = len(pop)
    results = get_fitness_results(pop, X, y, fitness_eval=fitness_eval, store_fitness='trainset_trainfit')
    ranked_index = get_ranked_index(results)
    partition = len(pop) - int(len(pop) * gap)
    top_i = ranked_index[-partition:]
    new_pop = [pop[i] for i in top_i]
    cleared_attrs = 0

    while len(new_pop) < env.pop_size:
        if len(new_pop) == (env.pop_size - 1):
            var_op = 0
        else:
            var_op = np.random.choice([0, 1], p=var_op_probs)

        if var_op == 0:
            parents_i = [np.random.choice(top_i)]
            progs = mutation([pop[parents_i[0]]])
        elif var_op == 1:
            parents_i = np.random.choice(top_i, 2, replace=False).tolist()
            progs = recombination([pop[i] for i in parents_i])
        new_pop += progs

        # if not cleared_attrs and fitness_eval.__name__ == 'fitness_sharing':
        #     for prog in progs:
        #         if getattr(prog, 'train_y_pred')[0] == -1:
        #             cleared_attrs = 1
        #             clear_attrs(new_pop)
        #             break

    return new_pop


#@profile
def check_fitness_after_variation(prog, instrs_changed):
    orig_eff_instrs = prog.effective_instrs
    effective_instrs = fit.find_introns(prog)

    if np.array_equal(effective_instrs, orig_eff_instrs) and set(instrs_changed).isdisjoint(orig_eff_instrs):
        pass
    else:
        clear_attrs([prog], clear_y_pred=True)


#@profile
def run_model(X, y, pop, selection, generations, X_test=None, y_test=None):
    start = time.time()
    filename_prefix = filenum()
    validation = (env.use_validation and env.use_subset)
    max_fitness_gen, max_fitness = 0, 0
    # Components to graph
    to_graph = {
        'top_trainfit_in_trainset': 1,  # Top training fitness value in training set
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

    if selection == const.Selection.STEADY_STATE_TOURN:
        select = tournament
    elif selection == const.Selection.BREEDER_MODEL:
        select = breeder
    else:
        raise ValueError('Invalid selection: {}'.format(selection))

    for i in range(env.generations):
        assert len(pop) == env.pop_size
        print('.', end='')
        sys.stdout.flush()

        # Re-sample from validation and training sets if using subsets
        if env.use_subset:
            X, y = env.subset_sampling(data, env.subset_size)
            # TODO: remove train examples from this? (use prev. valid data?)
            if validation:
                X_valid, y_valid = env.subset_sampling(data, env.validation_size)
        pop = select(X, y, pop, env.train_fitness_eval)

        # Run train/test fitness evaluations for data to be graphed
        if (i % env.graph_step == 0) or (i == (env.generations - 1)):
            if to_graph['top_trainfit_in_trainset'] or (env.train_fitness_eval == env.test_fitness_eval):
                trainset_results_with_trainfit = get_fitness_results(pop, X, y, fitness_eval=env.train_fitness_eval)
                top_trainfit_in_trainset.append(max(trainset_results_with_trainfit))
                trainfit_trainset_means.append(np.mean(trainset_results_with_trainfit))

            if (env.train_fitness_eval != env.test_fitness_eval) and (to_graph['top_test_fit_on_train']
                                                                      or to_graph['test_means']):
                if validation:
                    xvals, yvals = X_valid, y_valid
                else:
                    xvals, yvals = X, y
                trainset_results_with_testfit = get_fitness_results(pop, xvals, yvals,
                                                                    fitness_eval=env.test_fitness_eval)
                top_testfit_in_trainset.append(max(trainset_results_with_testfit))
                testfit_trainset_means.append(np.mean(trainset_results_with_testfit))

            if (X_test is not None and y_test is not None) and to_graph['top_train_prog_on_test']:
                if env.train_fitness_eval == env.test_fitness_eval:
                    trainset_results_with_testfit = trainset_results_with_trainfit
                testset_results_with_testfit = get_fitness_results([get_top_prog(pop, trainset_results_with_testfit)],
                                                                   X_test, y_test, fitness_eval=env.test_fitness_eval)[
                    0]
                top_prog_testfit_on_testset.append(testset_results_with_testfit)
                if testset_results_with_testfit > max_fitness:
                    max_fitness = testset_results_with_testfit
                    max_fitness_gen = i
            if to_graph['percentages']:
                cp = fit.class_percentages(get_top_prog(pop, trainset_results_with_testfit), X_test, y_test,
                                           data.classes)
                for p in cp:
                    percentages[p].append(cp[p])

        if ((env.graph_save_step is not None) and (i % env.graph_save_step == 0)) or (i == env.generations - 1):
            # Get graph parameters for graphing function - set to None to not display a graph component
            graphparam = [top_trainfit_in_trainset, trainfit_trainset_means, testfit_trainset_means,
                          top_prog_testfit_on_testset, top_testfit_in_trainset]
            graph_inc = [to_graph['top_trainfit_in_trainset'], to_graph['train_means'], to_graph['test_means'],
                         to_graph['top_train_prog_on_test'], to_graph['top_test_fit_on_train']]
            graphparam = list(map(lambda x: graphparam[x] if graph_inc[x] else None, range(len(graphparam))))
            graph(filename_prefix, env.graph_step, env.generations - 1, graphparam[0], graphparam[1], graphparam[2],
                  graphparam[3],
                  graphparam[4])

            if to_graph['percentages']:
                graph_percs(filename_prefix, env.graph_step, env.generations - 1, percentages)
    plt.close('all')

    # print_stats(pop, trainset_results_with_testfit, testset_results_with_testfit, top_prog_testfit_on_testset)
    print("Max fitness: {} at generation {}".format(max_fitness, max_fitness_gen))
    print("\nTime: {}".format(time.time() - start))
    return pop, trainset_results_with_trainfit, trainset_results_with_testfit, testset_results_with_testfit


def print_stats(pop, train_results_with_test_fit, test_results_with_test_fit, test_result_with_test_fit):
    print('\nTop program (from train_fitness on training set):')
    print('Test_fitness on test set: {} \nTest_fitness on train set: {}'.format(test_result_with_test_fit,
                                                                                max(train_results_with_test_fit)))

    top_prog = get_top_prog(pop, train_results_with_test_fit)
    train_cl = fit.class_percentages(top_prog, data.X_train, data.y_train, data.classes)
    test_cl = fit.class_percentages(top_prog, data.X_test, data.y_test, data.classes)
    print('Test class percentages: {}\nTrain class percentages (all training): {}'.format(test_cl, train_cl))

    test_fit_on_test = get_fitness_results(pop, data.X_test, data.y_test, fitness_eval=env.test_fitness_eval,
                                           store_fitness='testset_testfit')
    top_test_result = max(test_fit_on_test)
    top_result_i = test_fit_on_test.index(top_test_result)
    top_prog_from_test_on_train = get_fitness_results([pop[top_result_i]], data.X_train, data.y_train,
                                                      env.test_fitness_eval, store_fitness='trainset_testfit')[0]

    print('\nTop program (from test_fitness on test set):')
    print('Top test_fitness in test set: {} \nTest_fitness on train set for top: {}'.format(top_test_result,
                                                                                            top_prog_from_test_on_train))


def graph(n, graph_step, last_x, top_train_fit_on_train=None, train_means=None, test_means=None,
          top_train_prog_on_test=None, top_test_fit_on_train=None):
    gens = [i for i in range(env.generations) if (i % graph_step == 0)]
    if gens[-1] != last_x:
        gens.append(last_x)

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

    subset_str = ', Subset Size: {}'.format(env.subset_size) if env.use_subset else ''
    valid_str = ''
    if env.use_validation:
        valid_str = ', Validation Size: {}'.format(env.validation_size)
    op_str = ', '.join([const.OPS[x] for x in env.ops])
    plt.title('Data: {}\nSelection: {}\nPop Size: {}, Generations: {}, Step Size: {}{}{}\nTraining Fitness: {}, '
              'Test Fitness: {}\nOps: {}, Alpha: {}'.format(env.data_file, env.selection.value, env.pop_size,
                                                            env.generations,
                                                            env.graph_step, subset_str, valid_str,
                                                            env.train_fitness_eval.__name__,
                                                            env.test_fitness_eval.__name__, op_str, env.alpha))
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    plt.grid(which='both', axis='both')
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymax=1.02)
    ax.set_ylim(ymin=0)
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.tick_params(which='both', width=1)

    if TESTING == 0:
        filename = '{}{}_fitness.png'.format(const.IMAGE_DIR, n)
        fig.savefig(filename)
        print('Saved file: {}'.format(filename))


def graph_percs(n, graph_step, last_x, percentages):
    generations = [i for i in range(env.generations) if (i % graph_step == 0)]
    if generations[-1] != last_x:
        generations.append(last_x)

    fig = plt.figure(figsize=(13, 13), dpi=80)
    ax = fig.add_subplot(111)
    labels = sorted([perc for perc in percentages])
    for l in labels:
        ax.plot(generations, percentages[l], label=l)
    plt.title('% Classes Correct (Training data: {})'.format(env.data_file))
    plt.legend(bbox_to_anchor=(1.1, 1), fontsize=8)
    plt.grid(which='both', axis='both')
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymax=1.02)
    ax.set_ylim(ymin=0)
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.tick_params(which='both', width=1)

    if TESTING == 0:
        filename = '{}{}_classes.png'.format(const.IMAGE_DIR, n)
        fig.savefig(filename)
        print('Saved file: {}'.format(filename))


def filenum():
    with shelve.open('config') as c:
        try:
            c['num'] += 1
        except KeyError:
            c['num'] = 0
        n = c['num']
    return n


#@profile
def get_average_fitness(env, num_trials, train_fitness_eval, test_fitness_eval=None):
    if not test_fitness_eval:
        test_fitness_eval = train_fitness_eval

    max_test_fitness_on_train, mean_test_fitness_on_train, top_test_fitness_on_train, = [], [], []
    top_test_fitness, top_train_prog_on_test = [], []

    for i in range(num_trials):
        # New train/test data split for each trial if train/test data isn't pre-split
        try:
            const.TEST_DATA_FILES[env.data_file]
        except KeyError:
            env.load_data()

        print('Trial: {}'.format(i + 1))
        pop = gen_population(env.pop_size)
        pop, train_results_with_train_fit, train_results_with_test_fit, test_result_with_test_fit = run_model(
            data.X_train, data.y_train, pop,
            env.selection, env.generations,
            X_test=data.X_test,
            y_test=data.y_test)

        max_test_fitness_on_train.append(max(train_results_with_test_fit))
        mean_test_fitness_on_train.append(np.mean(train_results_with_test_fit))
        top_train_prog_on_test.append(test_result_with_test_fit)

        print('\nTop program (from train_fitness on training set):')
        print('Test_fitness on test set: {} \nTest_fitness on train set: {}'.format(test_result_with_test_fit,
                                                                                    max(train_results_with_test_fit)))

        top_prog = get_top_prog(pop, train_results_with_test_fit)
        train_cl = fit.class_percentages(top_prog, data.X_train, data.y_train, data.classes)
        test_cl = fit.class_percentages(top_prog, data.X_test, data.y_test, data.classes)
        print('Test class percentages: {}\nTrain class percentages (all training): {}'.format(test_cl, train_cl))

        test_fit_on_test = get_fitness_results(pop, data.X_test, data.y_test, fitness_eval=test_fitness_eval,
                                               store_fitness='testset_testfit')
        top_test_result = max(test_fit_on_test)
        top_result_i = test_fit_on_test.index(top_test_result)
        top_prog_from_test_on_train = \
        get_fitness_results([pop[top_result_i]], data.X_train, data.y_train, test_fitness_eval,
                            store_fitness='trainset_testfit')[0]
        top_test_fitness.append(top_test_result)
        top_test_fitness_on_train.append(top_prog_from_test_on_train)

        print('\nTop program (from test_fitness on test set):')
        print('Top test_fitness in test set: {} \nTest_fitness on train set for top: {}'.format(top_test_result,
                                                                                                top_prog_from_test_on_train))

    avg_max, avg_mean, avg_test = np.mean(max_test_fitness_on_train), np.mean(mean_test_fitness_on_train), np.mean(
        top_train_prog_on_test)
    avg_top_test, avg_top_on_train = np.mean(top_test_fitness), np.mean(top_test_fitness_on_train)

    print('\nTrials: {}\nAverage max test_fitness on train: {}\nAverage mean test_fitness on train: {}\n'
          'Average test_fitness on test: {}\n'.format(num_trials, avg_max, avg_mean, avg_test))
    print('Average top test_fitness on test: {}\n'
          'Average top test_fitness on train, from top test_fitness on test: {}\n'.format(avg_top_test,
                                                                                          avg_top_on_train))


def get_ranked_index(results):
    return [x[0] for x in sorted(enumerate(results), key=lambda i: i[1])]


def get_top_prog(pop, results):
    return pop[get_ranked_index(results)[-1]]


def print_info():
    print('Population size: {}\nGenerations: {}\nData: {}\nSelection Replacement: {}\n'
          'Alpha: {}\n'.format(env.pop_size, env.generations, env.data_file, env.selection.name, env.alpha))


#@profile
def main():
    trials = 1
    # get_average_fitness(env, trials, train_fitness_eval=env.train_fitness_eval, test_fitness_eval=env.test_fitness_eval)
    pop = gen_population(env.pop_size)
    run_model(data.X_train, data.y_train, pop,
              env.selection, env.generations,
              X_test=data.X_test,
              y_test=data.y_test)
    # pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    # pred = array('i', pred)
    # fit.fitness_sharing(pop, data.X_train, data.y_train)
    #all_y_pred = [fit.predicted_classes(prog, data.X_train) for prog in pop]
    # for prog in pop:
    #     prog.prog = np.array(prog.prog)
    # all_y_pred=vm.y_pred(np.asarray(pop), data.X_train)
    # vm.fitness_sharing(np.asarray(pop), data.X_train, pred)


def init_vm():
    vm.init(const.GEN_REGS, env.num_ipregs, env.output_dims)


env = Config()
data = Data()
if env.data_file:
    data.load_data(env)
    init_vm()

if __name__ == '__main__':
    main()
    #import cProfile
    #cProfile.run('main()', sort='time')

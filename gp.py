import const, random, vm, copy, sys, threading
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pdb
import matplotlib.pyplot as plt
from importlib import reload
from array import array

class Config:
    def __init__(self):
        self.num_ops = 4
        self.pop_size = 100
        self.generations = 500000
        self.graph_step = 10
        self.data_files = ['data/iris.data', 'data/tic-tac-toe.data', 'data/ann-train.data', 'data/shuttle.trn']
        self.data_file = self.data_files[0]
        self.data_by_classes = None
        self.standardize_method = const.StandardizeMethod.MEAN_VARIANCE
        self.selection = const.Selection.STEADY_STATE_TOURN
        self.alpha = 1
        self.use_subset = False
        self.subset_size = 200


    def load(self):
        self.train_fitness_eval = fitness_sharing
        self.test_fitness_eval = avg_detect_rate

        data = load_data(self.data_file)
        y = [ex[-1:len(ex)][0] for ex in data]
        self.classes = self.get_classes(y)
        y = [self.classes[label] for label in y]


        try:
            test_data = load_data(const.TEST_DATA_FILES[self.data_file])
            self.X_train = preprocess([ex[:len(ex)-1] for ex in data])
            self.y_train = y
            self.X_test = preprocess([ex[:len(ex)-1] for ex in test_data])
            self.y_test = [self.classes[label] for label in [ex[-1:len(ex)][0] for ex in test_data]]
        except KeyError:
            X = preprocess([ex[:len(ex)-1] for ex in data])
            X_train, X_test, y_train, y_test = split_data(X, y, env.standardize_method)
            self.X_train, self.X_test, self.y_train, self.y_test = preprocess(X_train), preprocess(X_test), y_train, y_test

        self.num_ipregs = len(self.X_train[0])
        self.output_dims = len(self.classes)
        self.max_vals = [const.GEN_REGS-1, max(const.GEN_REGS, self.num_ipregs)-1, self.num_ops-1, 1]

    def get_classes(self, data):
        classes = set(data)
        classmap = {}
        for i in range(len(classes)):
            classmap[classes.pop()] = i
        return classmap

class Prog:
    def __init__(self, prog):
        self.prog = prog
        self.effective_instrs = None
        # Fitness for training eval on train, testing eval on train, testing eval on test
        self.trainset_trainfit = None
        self.trainset_testfit = None
        self.testset_testfit = None
        self.train_y_pred = None

'''
Generating initial programs
'''
def gen_prog(prog_length, num_ipregs):
    prog = [[],[],[],[]]
    prog[const.TARGET] = np.random.randint(const.GEN_REGS, size=prog_length).tolist()
    prog[const.SOURCE] = np.random.randint(max(const.GEN_REGS, env.num_ipregs), size=prog_length).tolist()
    prog[const.OP] = np.random.randint(env.num_ops, size=prog_length).tolist()
    prog[const.MODE] = np.random.randint(2, size=prog_length).tolist()
    return Prog(prog)

def gen_population(pop_num, num_ipregs):
    pop = [gen_prog(const.PROG_LENGTH, env.num_ipregs) for n in range(0, pop_num)]
    for p in pop:
        p.effective_instrs = find_introns(p)
    return pop

'''
Initializing the data
'''
def load_data(fname, split=','):
    data = []
    with open(fname, 'r') as f:
        for line in f:
            data.append(line.split(split))
    return data

def standardize(data, method, alpha=1, vals=None):
    m = np.asmatrix(data)
    m_transpose = np.transpose(m)
    if method is const.StandardizeMethod.MEAN_VARIANCE:
        for col in range(len(data[0])):
            if vals:
                std = vals[0]
                mean = vals[1]
            else:
                std = np.std(m_transpose[col])
                mean = np.mean(m_transpose[col])
            for row in range(len(data)):
                m[row, col] = alpha*((m.item(row,col)-mean)/std)
        return m.tolist(), std, mean

    elif method is const.StandardizeMethod.LINEAR_TRANSFORM:
        for col in range(len(data[0])):
            if vals:
                min_x = vals[0]
                max_x = vals[1]
            else:
                max_x = max(m_transpose[col].tolist()[0])
                min_x = min(m_transpose[col].tolist()[0])
            for row in range(len(data)):
                m[row, col] = alpha*((m.item(row,col)-min_x)/(max_x-min_x))
        return m.tolist(), min_x, max_x
    else:
        raise AttributeError('Invalid standardize method')


def preprocess(data):
    for i in range(len(data)):
        try:
            data[i] = array('d', [float(x) for x in data[i]])
        except ValueError:
            preprocess(convert_non_num_data(data))
    return data

def convert_non_num_data(data):
    attrs = []
    for i in range (len(data)):
        attrs += [attr for attr in data[i]]
    attrs = list(set(attrs))
    for i in range(len(data)):
        data[i] = [attrs.index(attr)+1 for attr in data[i]]
    return data

def split_data(X, y, standardize_method, alpha=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    if standardize_method:
        X_train, val0, val1 = standardize(X_train, standardize_method, alpha=alpha)
        X_test = standardize(X_test, standardize_method, alpha=alpha, vals=[val0, val1])[0]
    return X_train, X_test, y_train, y_test

'''
Fitness evaluation functions
'''
def accuracy(prog, y, y_pred, store_fitness=None):
    acc = accuracy_score(y, y_pred)
    if store_fitness:
        setattr(prog, store_fitness, acc)
    return acc

#@profile
def avg_detect_rate(prog, y, y_pred, store_fitness=None):
    percentages = []
    for cl in set(y):
        cl_results = [i for i in range(len(y)) if y[i] == cl]
        percentages.append(sum([1 for i in cl_results if y[i] == y_pred[i]])/len(cl_results))
    fitness = np.mean(percentages)
    if store_fitness:
        setattr(prog, store_fitness, fitness)
    return fitness
#
# #@profile
# def fitness_sharing(pop, X, y, store_fitness=None):
#     training = (store_fitness == 'trainset_trainfit')
#     all_y_pred = [predicted_classes(prog, X, fitness_sharing=training) for prog in pop]
#     sum_fitness = 1+per_ex_fitness(y, all_y_pred)
#     all_fitness = [per_ex_fitness(y, [all_y_pred[i]])/sum_fitness for i in range(len(pop))]
#     return all_fitness
#@profile
def fitness_sharing(pop, X, y, store_fitness=None):
    #for p in pop:
        #print("Y_pred: {}".format(p.train_y_pred))
    training = (store_fitness == 'trainset_trainfit')
    all_y_pred = [predicted_classes(prog, X, fitness_sharing=training) for prog in pop]
    denoms = [sum([1 for j in range(len(all_y_pred)) if all_y_pred[j][i] == y[i]])+1 for i in range(len(y))]
    fitness = [sum([int(y_pred[i] == y[i])/denoms[i] for i in range(len(y))]) for y_pred in all_y_pred]
    return fitness

'''
Results
'''

# class progThread (threading.Thread):
#    def __init__(self, threadID):
#       threading.Thread.__init__(self)
#       self.threadID = threadID
#    def run(self):
#       print "Starting " + self.name
#       return predicted_classes(prog, X) if (prog.fitness is None or not training) else prog.fitness
#
# def print_time(threadName, counter, delay):
#    while counter:
#       if exitFlag:
#          threadName.exit()
#       time.sleep(delay)
#       print "%s: %s" % (threadName, time.ctime(time.time()))
#       counter -= 1

#@profile
def predicted_classes(prog, X, fitness_sharing=False):
    #pdb.set_trace()
    if fitness_sharing and prog.train_y_pred is not None:
        return prog.train_y_pred

    y_pred = [output.index(max(output)) for output in [vm.run_prog(prog, ex) for ex in X]]
    if fitness_sharing:
        prog.train_y_pred = y_pred
    return y_pred


class EvalProg(threading.Thread):
    def __init__(self, i, result_list, prog, X):
        threading.Thread.__init__(self)
        self.id = i
        self.result_list = result_list
        self.prog = prog
        self.X = X
    def run(self):
        y_pred = predicted_classes(self.prog, self.X)
        self.result_list[self.id] = y_pred


def caller_name(skip=2):
    """Get a name of a caller in the format module.class.method

       `skip` specifies how many levels of stack to skip while getting caller
       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

       An empty string is returned if skipped levels exceed stack height
    """
    stack = inspect.stack()
    start = 0 + skip
    if len(stack) < start + 1:
      return ''
    parentframe = stack[start][0]

    name = []
    module = inspect.getmodule(parentframe)
    # `modname` can be None when frame is executed directly in console
    # TODO(techtonik): consider using __main__
    if module:
        name.append(module.__name__)
    # detect classname
    if 'self' in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # XXX: there seems to be no way to detect static method call - it will
        #      be just a function call
        name.append(parentframe.f_locals['self'].__class__.__name__)
    codename = parentframe.f_code.co_name
    if codename != '<module>':  # top level usually
        name.append( codename ) # function or a method

    ## Avoid circular refs and frame leaks
    #  https://docs.python.org/2.7/library/inspect.html#the-interpreter-stack
    del parentframe, stack

    return ".".join(name)

#@profile
def get_fitness_results(pop, X, y, fitness_eval, store_fitness=None):
    #results = [fitness_eval(prog, y, predicted_classes(prog, X, v)) if (prog.fitness is None or not training) else prog.fitness for prog in pop]
    if fitness_eval == fitness_sharing:
        if (store_fitness is None) or (getattr(pop[0], store_fitness) is None):
            results = fitness_eval(pop, X, y, store_fitness=store_fitness)
        else:
            # TODO: When is this saved??
            results = [getattr(prog, store_fitness) for prog in pop]


    else:
        results = [fitness_eval(prog, y, predicted_classes(prog, X), store_fitness=store_fitness)
                   if (store_fitness is None or env.use_subset or getattr(prog, store_fitness) is None)
                   else getattr(prog, store_fitness) for prog in pop]
        # Speed testing:
        # results=[]
        # for prog in pop:
        #     if (store_fitness is None):
        #         c = predicted_classes(prog, X)
        #         fit = fitness_eval(prog, y, c)
        #         results.append(fit)
        #     elif getattr(prog, store_fitness) is None:
        #         c = predicted_classes(prog, X)
        #         fit = fitness_eval(prog, y, c,  store_fitness=store_fitness)
        #         results.append(fit)
        #     else:
        #         fit=getattr(prog, store_fitness)
        #         results.append(fit)
    return results

# def fitness_results(prog, X, y, fitness_eval, store_fitness=None, l=None):
#     if fitness_eval == fitness_sharing:
#         if (store_fitness is None) or (getattr(pop[0], store_fitness) is None):
#             results = fitness_eval(pop, X, y, store_fitness=store_fitness)
#         else:
#             results = [getattr(prog, store_fitness) for prog in pop]
#
#
#     else:
#         results = [fitness_eval(prog, y, predicted_classes(prog, X), store_fitness=store_fitness)
#                    if (store_fitness is None or env.use_subset or getattr(prog, store_fitness) is None)
#                    else getattr(prog, store_fitness) for prog in pop]
#
#
# from multiprocessing import Process, Value, Array
# def get_fitness_results(pop, X, y, fitness_eval, store_fitness=None):
#      arr = Array('d', [0]*len(pop))
#
#     if fitness_eval == fitness_sharing:
#         if (store_fitness is None) or (getattr(pop[0], store_fitness) is None):
#             results = fitness_results(pop, X, y, fitness_eval, store_fitness=store_fitness, l=)
#         else:
#             results = [getattr(prog, store_fitness) for prog in pop]
#
#
#     else:
#         arr = [get_attr(prog, store_fitness) if not (store_fitness is None or env.use_subset or getattr(prog, store_fitness) is None) else 0 for prog in pop]
#         results = [fitness_eval(prog, y, predicted_classes(prog, X), store_fitness=store_fitness)
#                    if (store_fitness is None or env.use_subset or getattr(prog, store_fitness) is None)
#                    else getattr(prog, store_fitness) for prog in pop]
#     return results




'''
Variation operators
'''
#@profile
# Recombination - swap sections of 2 programs
def recombination(progs):
    progs = copy.deepcopy(progs)
    assert len(progs) == 2
    prog0, prog1 = progs[0].prog, progs[1].prog
    prog_len = len(prog0[0])
    start_index = random.randint(0, prog_len-2)
    end_limit = prog_len-1 if start_index is 0 else prog_len # avoid swapping whole program
    end_index = random.randint(start_index+1, end_limit)

    for col in range(len(prog0)):
        prog1[col][start_index:end_index], prog0[col][start_index:end_index] = prog0[col][start_index:end_index], prog1[col][start_index:end_index]
    for prog in progs:
        check_fitness_after_variation(prog, list(range(start_index, end_index)))
    return [progs[0], progs[1]]

#@profile
# Mutation - change 1 value in the program
def mutation(progs, effective_mutations=True):
    step_size = 1
    min_lines, max_lines = 1, len(progs[0].prog[0])

    progs = copy.deepcopy(progs)
    children = []
    for prog in progs:
        # Test - effective mutations
        if effective_mutations:
            if prog.effective_instrs is None:
                find_introns(prog)
            num_lines = random.randint(min_lines, min(max_lines, len(prog.effective_instrs)))
            lines = np.random.choice(prog.effective_instrs, size=num_lines, replace=False)
        else:
            num_lines = random.randint(min_lines, max_lines)
            lines = np.random.choice(list(range(max_lines)), size=num_lines, replace=False)
        for index in lines:
            col = random.randint(0, len(prog.prog)-1)
            orig_val = prog.prog[col][index]
            new_val = orig_val
            if col == const.OP:
                options = list(range(env.max_vals[const.OP]+1))
                options.pop(orig_val)
                new_val = np.random.choice(options)
            else:
                while (new_val == orig_val):
                    if new_val == 0:
                        op = 0
                    elif (new_val == env.max_vals[col]):
                        op = 1
                    else:
                        op = random.randint(0, 1)
                    new_val = (orig_val+step_size) if op is 0 else (orig_val-step_size)
            prog.prog[col][index] = new_val
        check_fitness_after_variation(prog, lines)
        children.append(prog)
    return children

'''
Selection
'''
#@profile
# Steady state tournament for selection
def tournament(X, y, pop, fitness_eval, var_op_probs=[0.5,0.5]):
    indivs = set()
    while len(indivs) < const.TOURNAMENT_SIZE:
        indivs.add(random.randint(0, len(pop)-1))
    selected_i = list(indivs)
    results = get_fitness_results([pop[i] for i in selected_i], X, y, fitness_eval=fitness_eval, store_fitness='trainset_trainfit')

    winners_i = []
    while len(winners_i) < 2:
        max_i = results.index(max(results))
        winners_i.append(selected_i[max_i])
        results[max_i] = -float('inf')
    losers_i = [i for i in selected_i if i not in winners_i]
    parents = [pop[i] for i in winners_i]
    for i in sorted(losers_i, reverse=True):
        del pop[i]

    var_op = np.random.choice([0,1], p=var_op_probs)
    if var_op == 0:
        progs = mutation(parents)
    elif var_op == 1:
        progs = recombination(parents)

    for p in progs:
        p.orig=False
    pop += progs

    if fitness_eval == fitness_sharing:
        for prog in progs:
            if getattr(prog, 'trainset_trainfit') is None:
                clear_attrs(pop)
                break

    return pop

# Breeder model for selection
def breeder(X, y, pop, fitness_eval, gap=0.2, var_op_probs=[0.5,0.5]):
    env.pop_size = len(pop)
    results = get_fitness_results(pop, X, y, fitness_eval=fitness_eval, store_fitness='trainset_trainfit')
    ranked_index = get_ranked_index(results)

    partition = len(pop) - int(len(pop)*gap)
    top_i = ranked_index[-partition:]
    new_pop = [pop[i] for i in top_i]

    while len(new_pop) < env.pop_size:
        if len(new_pop) == (env.pop_size-1):
            var_op = 0
        else:
            var_op = np.random.choice([0,1], p=var_op_probs)

        if var_op == 0:
            parents_i = [np.random.choice(top_i)]
            progs = mutation([pop[parents_i[0]]])
        elif var_op == 1:
            parents_i = np.random.choice(top_i, 2, replace=False).tolist()
            progs = recombination([pop[i] for i in parents_i])
        new_pop += progs

        if fitness_eval == fitness_sharing:
            for prog in progs:
                if getattr(prog, 'trainset_trainfit') is None:
                    clear_attrs(pop)
                    break

    return new_pop

def clear_attrs(progs, clear_y_pred=False):
    for prog in progs:
        prog.trainset_trainfit = None
        prog.trainset_testfit = None
        prog.testset_testfit = None

        if clear_y_pred:
            prog.train_y_pred = None

#@profile
def check_fitness_after_variation(prog, instrs_changed):
    orig_effective_instrs = prog.effective_instrs
    effective_instrs = find_introns(prog)

    if np.array_equal(effective_instrs, orig_effective_instrs) and set(instrs_changed).isdisjoint(orig_effective_instrs):
        pass
    else:
        clear_attrs([prog], clear_y_pred=True)

#@profile
def run_model(X, y, pop, selection, generations, fitness_eval, X_test=None, y_test=None, show_graph=1):
    to_graph = {
        'top_trainfit_in_trainset': 1,
        'train_means': 1,
        'test_means': 1,
        'top_train_prog_on_test': 1,
        'top_test_fit_on_train': 1
    }
    graph_step = env.graph_step
    train_fitness_eval = env.train_fitness_eval
    test_fitness_eval = env.test_fitness_eval
    trainset_results_with_trainfit, trainset_results_with_testfit, testset_results_with_testfit = [], [], []
    trainfit_trainset_means, testfit_trainset_means, top_trainfit_in_trainset, top_prog_testfit_on_testset, top_testfit_in_trainset = [], [], [], [], []
    print_info()

    # print('Start fitness: ')
    # results = get_fitness_results(pop, X, y, env.test_fitness_eval)
    # print('{}:  max({}): {}\tmean:{}'.format(1, results.index(max(results)), max(results), (sum(results) / float(len(results)))))
    if selection == const.Selection.STEADY_STATE_TOURN:
        select = tournament
    elif selection == const.Selection.BREEDER_MODEL:
        select = breeder
    else:
        raise ValueError('Invalid selection: {}'.format(selection))

    for i in range(env.generations):
        print('.', end='')
        sys.stdout.flush()
        assert len(pop) == env.pop_size
        pop = select(X, y, pop, fitness_eval)

        if env.use_subset:
            X, y = get_data_subset(X, y)

        if (i % graph_step == 0) or (i == (env.generations-1)):
            if to_graph['top_trainfit_in_trainset'] or to_graph['train_means'] or (env.train_fitness_eval == env.test_fitness_eval):
                trainset_results_with_trainfit = get_fitness_results(pop, X, y, fitness_eval=env.train_fitness_eval, store_fitness='trainset_trainfit')
                top_trainfit_in_trainset.append(max(trainset_results_with_trainfit))
                trainfit_trainset_means.append(np.mean(trainset_results_with_trainfit))

            if (env.train_fitness_eval != env.test_fitness_eval) and (to_graph['top_test_fit_on_train'] or to_graph['test_means']):
                trainset_results_with_testfit = get_fitness_results(pop, X, y, fitness_eval=test_fitness_eval, store_fitness='trainset_testfit')
                top_testfit_in_trainset.append(max(trainset_results_with_testfit))
                testfit_trainset_means.append(np.mean(trainset_results_with_testfit))

            if (X_test and y_test) and to_graph['top_train_prog_on_test']:
                if env.train_fitness_eval == env.test_fitness_eval:
                    trainset_results_with_testfit = trainset_results_with_trainfit
                testset_results_with_testfit = get_fitness_results([get_top_prog(pop, trainset_results_with_testfit)], X_test, y_test, fitness_eval=test_fitness_eval, store_fitness='testset_testfit')[0]
                top_prog_testfit_on_testset.append(testset_results_with_testfit)

    # print('End fitness: ')
    # results = get_fitness_results(pop, X, y, env.test_fitness_eval)
    # print('{}:  max({}): {}\tmean:{}'.format(1, results.index(max(results)), max(results), sum(results) / float(len(results))))
    if show_graph:
        graph_params = [top_trainfit_in_trainset, trainfit_trainset_means, testfit_trainset_means, top_prog_testfit_on_testset, top_testfit_in_trainset]
        graph_inc = [to_graph['top_trainfit_in_trainset'], to_graph['train_means'], to_graph['test_means'], to_graph['top_train_prog_on_test'], to_graph['top_test_fit_on_train']]
        graph_params = list(map(lambda x: graph_params[x] if graph_inc[x] else None, range(len(graph_params))))
        graph(graph_step, env.generations-1, graph_params[0], graph_params[1], graph_params[2], graph_params[3], graph_params[4])
    return pop, trainset_results_with_trainfit, trainset_results_with_testfit, testset_results_with_testfit

def get_data_subset(X, y):
    if not env.data_by_classes:
        env.data_by_classes = []
        for cl in env.classes.values():
            env.data_by_classes.append([X[i] for i in range(len(X)) if y[i] == cl])
    subset_size = int(env.subset_size / len(env.classes))
    subset, subset_cl = [], []
    for i in range(len(env.data_by_classes)):
        class_size = len(env.data_by_classes[i])
        if class_size <= subset_size:
            subset += env.data_by_classes[i]
            subset_cl += [i]*class_size
        else:
            subset += trainset_testfit_split(env.data_by_classes[i], train_size=(subset_size/class_size))[0]
            subset_cl += [i]*subset_size
    return subset, subset_cl

def graph(graph_step, last_x, top_results=None, train_means=None, test_means=None, top_train_prog_on_test=None, top_test_fit_on_train=None):
    generations = [i for i in range(env.generations) if (i % graph_step == 0)]
    if generations[-1] != last_x:
        generations.append(last_x)

    fig = plt.figure(figsize=(13, 13), dpi=80)
    ax = fig.add_subplot(111)
    labels = [
        'Max Train Fitness in Train Set', 'Mean Train Fitness in Train Set', 'Mean Test Fitness in Train Set',
        'Best Train Prog on Test Set (using test_fit)', 'Best Train Prog on Train Set (using test_fit)'
    ]
    if top_results:
        ax.plot(generations, top_results, label=labels[0])
    if train_means:
        ax.plot(generations, train_means, label=labels[1])
    if test_means:
        ax.plot(generations, test_means, label=labels[2])
    if top_train_prog_on_test:
        ax.plot(generations, top_train_prog_on_test, label=labels[3])
    if top_test_fit_on_train:
        ax.plot(generations, top_test_fit_on_train, label=labels[4])
    plt.title('Data: {}\nPop Size: {}, Generations: {}, Step Size: {}\nTraining Fitness: {}, Test Fitness: {}'.format(
        env.data_file, env.pop_size, env.generations, env.graph_step, env.train_fitness_eval.__name__,env.test_fitness_eval.__name__))
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    plt.grid(which='major', axis='both')
    ax.set_xlim(xmin=0)
    plt.show()

#@profile
def get_average_fitness(env, num_trials, train_fitness_eval, test_fitness_eval=None, validation=False):
    if not test_fitness_eval:
        test_fitness_eval = train_fitness_eval
    training = (train_fitness_eval == test_fitness_eval)
    max_test_fitness_on_train , mean_test_fitness_on_train, top_train_prog_on_test, valid_test_fitness, top_test_fitness_on_train = [], [], [], [], []
    top_test_fitness, valid_train_fitness = [], []

    # Fix for already split data
    if validation:
        X_train, X_valid, y_train, y_valid = split_data(X, y, None)
    else:
        X_train, y_train = env.X_train, env.y_train
    X_test, y_test = env.X_test, env.y_test

    for i in range(num_trials):
        print('Trial: {}'.format(i+1))
        pop = gen_population(env.pop_size, env.num_ipregs)
        pop, train_results_with_train_fit, train_results_with_test_fit, test_result_with_test_fit = run_model(X_train, y_train, pop,
                                                                                                              env.selection, env.generations,
                                                                                                              train_fitness_eval,
                                                                                                              X_test=X_test, y_test=y_test)
        # Testing
        env.pop = pop
        env.X_train, env.X_test = X_train, X_test
        env.y_train, env.y_test = y_train, y_test

        max_test_fitness_on_train.append(max(train_results_with_test_fit))
        mean_test_fitness_on_train.append(np.mean(train_results_with_test_fit))
        top_train_prog_on_test.append(test_result_with_test_fit)

        print('\nTop program (from train_fitness on training set):')
        print('Test_fitness on test set: {} \nTest_fitness on train set: {}'.format(test_result_with_test_fit, max(train_results_with_test_fit)))

        top_prog = get_top_prog(pop, train_results_with_test_fit)
        train_cl = class_percentages(top_prog, X_train, y_train, env.classes)
        test_cl = class_percentages(top_prog, X_test, y_test, env.classes)
        print('Test class percentages: {}\nTrain class percentages: {}'.format(test_cl, train_cl))

        test_fit_on_test = get_fitness_results(pop, X_test, y_test, fitness_eval=test_fitness_eval, store_fitness='testset_testfit')
        top_test_result = max(test_fit_on_test)
        top_result_i = test_fit_on_test.index(top_test_result)
        top_prog_from_test_on_train = get_fitness_results([pop[top_result_i]], X_train, y_train, test_fitness_eval, store_fitness='trainset_testfit')[0]
        top_test_fitness.append(top_test_result)
        top_test_fitness_on_train.append(top_prog_from_test_on_train)

        print('\nTop program (from test_fitness on test set):')
        print('Top test_fitness in test set: {} \nTest_fitness on train set for top: {}'.format(top_test_result, top_prog_from_test_on_train))

        if validation:
            valid_fitness_eval = fitness_eval
            results = get_fitness_results(pop, X_valid, y_valid, fitness_eval=valid_fitness_eval)
            top_prog = get_top_prog(pop, results)
            test_results = get_fitness_results([top_prog], X_test, y_test, valid_fitness_eval)
            valid_score = get_fitness_results([top_prog], X_train, y_train, fitness_eval=valid_fitness_eval)

            test_acc = get_fitness_results([top_prog], X_test, y_test, fitness_eval=avg_detect_rate)[0]
            train_acc = get_fitness_results([top_prog], X_train, y_train, fitness_eval=avg_detect_rate)[0]
            valid_test_fitness.append(test_acc)
            valid_train_fitness.append(train_acc)
            print('Top fitness on validation data: {} (Acc: {}) \nScore on train data for top: {} (Acc: {})'.format(test_results[0], test_acc, valid_score, train_acc))

    avg_max, avg_mean, avg_test = np.mean(max_test_fitness_on_train), np.mean(mean_test_fitness_on_train), np.mean(top_train_prog_on_test)
    avg_top_test, avg_top_on_train = np.mean(top_test_fitness), np.mean(top_test_fitness_on_train)

    print('\nTrials: {}\nAverage max test_fitness on train: {}\nAverage mean test_fitness on train: {}\nAverage test_fitness on test: {}\n'.format(num_trials, avg_max, avg_mean, avg_test))
    print('Average top test_fitness on test: {}\nAverage top test_fitness on train, from top test_fitness on test: {}\n'.format(avg_top_test, avg_top_on_train))
    if validation:
        avg_valid_test, avg_top_valid_on_train = np.mean(valid_test_fitness), np.mean(valid_train_fitness)
        print('Average test acc from top valid: {}\nAverage top valid program acc on training: {}\n'.format(avg_valid_test, avg_top_valid_on_train))


def get_ranked_index(results):
    return [x[0] for x in sorted(enumerate(results), key=lambda i:i[1])]

def get_top_prog(pop, results):
    return pop[get_ranked_index(results)[-1]]

def run_top_prog(results, X_test, y_test, pop, fitness_eval):
    top_prog = get_top_prog(pop, results)
    results0 = get_fitness_results([top_prog], X_test, y_test, fitness_eval)
    results1 = get_fitness_results([top_prog], X_train, y_train, fitness_eval)
    print('Fitness on test data: {}\nFitness on orig data: {}'.format(results0[0], results1[0]))
    all_results = get_fitness_results(pop, X_test, y_test, fitness_eval)
    top_result = max(all_results)
    top_result_i = all_results.index(max(all_results))
    pop_score = get_fitness_results([pop[top_result_i]], X_train, y_train, fitness_eval, training=True)
    print('Top fitness on test data: {} \nScore on orig data for top: {}'.format(top_result, pop_score))
    return top_prog

def find_introns(prog):
    instrs = vm.find_introns(prog.prog[const.TARGET], prog.prog[const.SOURCE], prog.prog[const.MODE])
    prog.effective_instrs = instrs
    return instrs

def class_percentages(prog, X, y, classes):
    percentages = {}
    y_pred = predicted_classes(prog, X)

    for cl in classes:
        cl_results = [i for i in range(len(y)) if y[i] == classes[cl]]
        perc = sum([1 for i in cl_results if y[i] == y_pred[i]])/len(cl_results)
        percentages[cl] = perc
    return percentages

def print_info():
    print('Population size: {}\nGenerations: {}\nData: {}\nSelection Replacement: {}\nAlpha: {}\n'.format(env.pop_size, env.generations, env.data_file, env.selection.name, env.alpha))

#@profile
def main():
    trials = 1
    get_average_fitness(env, trials, train_fitness_eval=env.train_fitness_eval, test_fitness_eval=env.test_fitness_eval)

def init_vm():
    vm.init(const.GEN_REGS, env.num_ipregs, env.num_ops, env.output_dims)

env = Config()
env.load()
init_vm()

if __name__ == '__main__':

    main()

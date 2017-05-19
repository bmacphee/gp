import const, random, vm, copy, sys, threading
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pdb
import matplotlib.pyplot as plt
from importlib import reload

class Config:
    def __init__(self):
        self.num_ops = 4
        self.pop_size = 100
        self.generations = 50
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
        self.train_train = None
        self.train_test = None
        self.test_test = None
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

from array import array
def preprocess(data):
    print('preprocess')
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

@profile
def avg_detect_rate(prog, y, y_pred, store_fitness=None):
    percentages = []
    for cl in set(y):
        cl_results = [i for i in range(len(y)) if y[i] == cl]
        percentages.append(sum([1 for i in cl_results if y[i] == y_pred[i]])/len(cl_results))
    fitness = sum(percentages) / float(len(percentages))
    if store_fitness:
        setattr(prog, store_fitness, fitness)
    return fitness

@profile
def fitness_sharing(pop, X, y, store_fitness=None):
    training = (store_fitness == 'train_train')
    all_y_pred = [predicted_classes(prog, X, fitness_sharing=training) for prog in pop]
    sum_fitness = 1+per_ex_fitness(y, all_y_pred)
    all_fitness = [per_ex_fitness(y, [all_y_pred[i]])/sum_fitness for i in range(len(pop))]
    return all_fitness

def per_ex_fitness(y, y_preds):
    return sum([len([1 for i in range(len(y)) for pred in y_preds if y[i] == pred[i]])])

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

import threading

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

@profile
def get_fitness_results(pop, X, y, fitness_eval, store_fitness=None):
    #results = [fitness_eval(prog, y, predicted_classes(prog, X, v)) if (prog.fitness is None or not training) else prog.fitness for prog in pop]
    if fitness_eval == fitness_sharing:
        if (store_fitness is None) or (getattr(pop[0], store_fitness) is None):
            results = fitness_eval(pop, X, y, store_fitness=store_fitness)
        else:
            results = [getattr(prog, store_fitness) for prog in pop]


    else:
        # results = [fitness_eval(prog, y, predicted_classes(prog, X), store_fitness=store_fitness)
        #            if (store_fitness is None or env.use_subset or getattr(prog, store_fitness) is None)
        #            else getattr(prog, store_fitness) for prog in pop]
        results=[]
        for prog in pop:
            if (store_fitness is None):
                c = predicted_classes(prog, X)
                fit = fitness_eval(prog, y, c)
                results.append(fit)
            elif getattr(prog, store_fitness) is None:
                c = predicted_classes(prog, X)
                fit = fitness_eval(prog, y, c,  store_fitness=store_fitness)
                results.append(fit)
            else:
                fit=getattr(prog, store_fitness)
                results.append(fit)
    return results

# def get_fitness_results(pop, X, y, fitness_eval, training=False, store_test_fitness=False):
#     pop_size = len(pop)
#     y_pred_results = [0]*pop_size
#
#     if fitness_eval == fitness_sharing:
#         results = [0]*pop_size
#     else:
#         if not store_test_fitness:
#             results = [0 if (pop[i].fitness is None or not training or env.use_subset) else prog[i].fitness for i in range(pop_size)]
#         else:
#             results = [0 if (pop[i].acc_fitness is None or env.use_subset) else prog[i].acc_fitness for i in range(pop_size)]
#
#     threads = [EvalProg(i, y_pred_results, pop[i], X) for i in range(pop_size) if results[i] == 0]
#
#     for thread in threads:
#         thread.start()
#     for thread in threads:
#         thread.join()
#
#     if fitness_eval == fitness_sharing:
#         results = fitness_eval(pop, X, y, training)
#     else:
#         for i in np.nonzero(y_pred_results)[0]:
#             results[i] = fitness_eval(pop[i], y, y_pred_results[i], training=training, store_test_fitness=store_test_fitness)
#     return results

@profile
def predicted_classes(prog, X, fitness_sharing=False):
    #pdb.set_trace()
    if fitness_sharing and prog.train_y_pred is not None:
        return prog.train_y_pred

    y_pred = []
    for ex in X:
        output = vm.run_prog(prog, ex)
        y_pred.append(output.index(max(output)))
    if fitness_sharing:
        prog.train_y_pred = y_pred
    return y_pred

'''
Variation operators
'''
@profile
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
@profile
# Mutation - change 1 value in the program
def mutation(progs, effective_mutations=True):
    step_size = 1
    progs = copy.deepcopy(progs)
    children = []
    for prog in progs:
        # Test - effective mutations
        if effective_mutations:
            if prog.effective_instrs is None:
                prog.effective_instrs = find_introns(prog)
            index = np.random.choice(prog.effective_instrs)
        else:
            index = random.randint(0, len(prog.prog[0])-1)
        col = random.randint(0, len(prog.prog)-1)
        orig_val = prog.prog[col][index]
        new_val = orig_val
        if col == const.OP:
            options = list(range(env.max_vals[const.OP]+1))
            options.pop(orig_val)
            new_val = np.random.choice(options)
        else:
            op = random.randint(0, 1)
            while (new_val == orig_val):
                new_val = (orig_val+step_size) if op is 0 else (orig_val-step_size)
                if (new_val < 0) or (new_val > env.max_vals[col]):
                    new_val = orig_val
                    op = 0 if op is 1 else 1
        prog.prog[col][index] = new_val
        check_fitness_after_variation(prog, [index])
        children.append(prog)
    return children

'''
Selection
'''
@profile
# Steady state tournament for selection
def tournament(X, y, pop, fitness_eval, var_op_probs=[0.5,0.5]):
    indivs = set()
    while len(indivs) < const.TOURNAMENT_SIZE:
        indivs.add(random.randint(0, len(pop)-1))
    selected_i = list(indivs)
    results = get_fitness_results([pop[i] for i in selected_i], X, y, fitness_eval=fitness_eval, store_fitness='train_train')

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

    pop += progs

    if fitness_eval == fitness_sharing:
        for prog in progs:
            if getattr(prog, 'train_train') is None:
                clear_attrs(pop)
                break

    return pop

# Breeder model for selection
def breeder(X, y, pop, fitness_eval, gap=0.2, var_op_probs=[0.5,0.5]):
    env.pop_size = len(pop)
    results = get_fitness_results(pop, X, y, fitness_eval=fitness_eval, store_fitness='train_train')
    ranked_index = get_ranked_index(results)

    partition = int(len(pop)*gap)
    top_i = ranked_index[-partition:]
    new_pop = [pop[i] for i in top_i]
    num_children = env.pop_size - len(new_pop)
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
                if getattr(prog, 'train_train') is None:
                    clear_attrs(pop)
                    break

    return new_pop

def clear_attrs(progs, clear_y_pred=False):
    for prog in progs:
        prog.train_train = None
        prog.train_test = None
        prog.test_test = None

        if clear_y_pred:
            prog.train_y_pred = None
@profile
def check_fitness_after_variation(prog, instrs_changed):
    effective_instrs = vm.find_introns(prog.prog[const.TARGET], prog.prog[const.SOURCE], prog.prog[const.MODE])
    #if (effective_instrs == prog.effective_instrs) and set(instrs_changed).isdisjoint(effective_instrs):
    if np.array_equal(effective_instrs, prog.effective_instrs) and set(instrs_changed).isdisjoint(effective_instrs):
        pass
    else:
        clear_attrs([prog], clear_y_pred=True)
        prog.effective_instrs = effective_instrs

@profile
def run_model(X, y, pop, selection, generations, fitness_eval, X_test=None, y_test=None):
    print_info()
    train_results_with_train_fit, train_results_with_test_fit, test_results_with_test_fit = [], [], []
    train_fit_means, top_train_fit_on_train, top_on_test, best_train = [], [], [], []
    train_fitness_eval = env.train_fitness_eval
    test_fitness_eval = env.test_fitness_eval
    print('Start fitness: ')
    results = get_fitness_results(pop, X, y, env.test_fitness_eval)
    print('{}:  max({}): {}\tmean:{}'.format(1, results.index(max(results)), max(results), (sum(results) / float(len(results)))))
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

        train_results_with_train_fit = get_fitness_results(pop, X, y, fitness_eval=env.train_fitness_eval, store_fitness='train_train')
        top_train_fit_on_train.append(max(train_results_with_train_fit))
        train_fit_means.append(sum(train_results_with_train_fit) / float(len(train_results_with_train_fit)))

        if env.train_fitness_eval != env.test_fitness_eval:
            train_results_with_test_fit = get_fitness_results(pop, X, y, fitness_eval=test_fitness_eval, store_fitness='train_test')
            best_train.append(max(train_results_with_test_fit))

        if X_test and y_test:
            test_results_with_test_fit = get_fitness_results([get_top_prog(pop, train_results_with_test_fit)], X_test, y_test, fitness_eval=test_fitness_eval, store_fitness='test_test')[0]
            top_on_test.append(test_results_with_test_fit)
        # if not (train_fitness_eval == avg_detect_rate):
        #     results = get_fitness_results(pop, X, y, fitness_eval=avg_detect_rate, store_test_fitness=True)
        #     top_acc.append(max(results))
    print('End fitness: ')
    results = get_fitness_results(pop, X, y, env.test_fitness_eval)
    print('{}:  max({}): {}\tmean:{}'.format(1, results.index(max(results)), max(results), sum(results) / float(len(results))))

    #graph(top_train_fit_on_train, train_fit_means, top_on_test, best_train)
    return pop, train_results_with_train_fit, train_results_with_test_fit, test_results_with_test_fit

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
            subset += train_test_split(env.data_by_classes[i], train_size=(subset_size/class_size))[0]
            subset_cl += [i]*subset_size
    return subset, subset_cl

def graph(top_results=None, means=None, top_train_prog_on_test=None, top_test_fit_on_train=None):
    generations = list(range(env.generations))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = ['Max Train Fitness on Train', 'Mean Train Fitness on Train', 'Best Train Prog on Test (using test_fit)', 'Best Train Prog on Train (using test_fit)']
    if top_results:
        ax.plot(generations, top_results, label=labels[0])
    if means:
        ax.plot(generations, means, label=labels[1])
    if top_train_prog_on_test:
        ax.plot(generations, top_train_prog_on_test, label=labels[2])
    if top_test_fit_on_train:
        ax.plot(generations, top_test_fit_on_train, label=labels[3])
    plt.title('Data: {}\nPop Size: {}, Generations: {}\nTraining Fitness: {}, Test Fitness: {}'.format(env.data_file, env.pop_size, env.generations, env.train_fitness_eval.__name__,env.test_fitness_eval.__name__))
    plt.legend(loc='best', fontsize=7)
    #plt.legend(bbox_to_anchor=(1,1), loc='lower center', ncol=1)
    plt.grid()
    ax.set_xlim(xmin=0)
    plt.show()

@profile
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
        pop = gen_population(env.pop_size, env.num_ipregs)
        pop, train_results_with_train_fit, train_results_with_test_fit, test_result_with_test_fit = run_model(X_train, y_train, pop, env.selection, env.generations, train_fitness_eval, X_test=X_test, y_test=y_test)
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

        test_fit_on_test = get_fitness_results(pop, X_test, y_test, fitness_eval=test_fitness_eval, store_fitness='test_test')
        top_test_result = max(test_fit_on_test)
        top_result_i = test_fit_on_test.index(top_test_result)
        top_prog_from_test_on_train = get_fitness_results([pop[top_result_i]], X_train, y_train, test_fitness_eval, store_fitness='train_test')[0]
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
    print('Average top test_fitness in test: {}\nAverage top test_fitness on train, from top test_fitness on test: {}\n'.format(avg_top_test, avg_top_on_train))
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
    return vm.find_introns(prog.prog[const.TARGET], prog.prog[const.SOURCE], prog.prog[const.MODE])

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

@profile
def main():
    #X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(env.X, env.y, env.standardize_method, validation=False)
    trials = 1
    get_average_fitness(env, trials, train_fitness_eval=env.train_fitness_eval, test_fitness_eval=env.test_fitness_eval)

def init_vm():
    vm.init(const.GEN_REGS, env.num_ipregs, env.num_ops, env.output_dims)
    # vm.num_genregs = const.GEN_REGS
    # vm.num_ipregs = env.num_ipregs
    # vm.num_ops = env.num_ops
    # vm.output_dims = env.output_dims

env = Config()
env.load()
init_vm()
#v = vm.Vm(const.GEN_REGS, env.num_ipregs, env.num_ops, env.output_dims)

#pop = gen_population(env.pop_size, env.num_ipregs)
# get_average_fitness(env, 1, train_fitness_eval=env.train_fitness_eval, test_fitness_eval=env.test_fitness_eval)

if __name__ == '__main__':

    main()
    #X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(env.X, env.y, env.standardize_method, validation=True)
    # pop = run_model(X_train, y_train, v, pop, selection, env.generations)
    # results = get_fitness_results(pop, X_train, y_train, v, fitness_eval=fitness_eval)
    # top_prog = run_top_prog(results, X_test, y_test, v, pop)
    #
    # pop = run_model(X_train, y_train, v, pop, selection, env.generations)
    # results = get_fitness_results(pop, X_valid, y_valid, v, fitness_eval=fitness_eval)
    # top_prog = run_top_prog(results, X_test, y_test, v, pop)
    # print('\n\nNo validation:')
    # results = get_fitness_results(pop, X_train, y_train, v, fitness_eval=fitness_eval)
    # top_prog = run_top_prog(results, X_test, y_test, v, pop)


    #results = run_model(X_train, y_train, v, pop, selection, env.generations)



    #gp.get_average_fitness(gp.env.X, gp.env.y, gp.v, 1, fitness_eval=gp.env.fitness_eval, validation=True)
    #run_top_prog(results, X_test, y_test, v, pop)

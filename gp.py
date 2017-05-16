import const, random, vm, copy, sys
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pdb
import matplotlib.pyplot as plt


#input
# env.num_ops = 4
# env.pop_size = 1000
# env.generations = 20

# data_files = ['data/iris.data', 'data/tic-tac-toe.data']
# data_file = data_files[0]
# standardize_method = const.StandardizeMethod.MEAN_VARIANCE
# selection = const.Selection.STEADY_STATE_TOURN
# alpha = 1

class Config:
    def __init__(self):
        self.num_ops = 4
        self.pop_size = 50
        self.generations = 50
        self.data_files = ['data/iris.data', 'data/tic-tac-toe.data', 'data/ann-train.data', 'data/shuttle.trn']
        self.test_data_files = ['data/ann-test.data', 'data/shuttle.tst']
        self.data_file = self.data_files[0]
        self.test_data_file = None
        self.standardize_method = const.StandardizeMethod.MEAN_VARIANCE
        self.selection = const.Selection.STEADY_STATE_TOURN
        self.alpha = 1

    def load(self):
        self.var_ops = [recombination, mutation]
        self.fitness_eval = avg_detect_rate
        data = load_data(self.data_file)
        if self.test_data_file:
            test_data = load_data(self.test_data_file)
        else:
            self.X = preprocess([ex[:len(ex)-1] for ex in data])
            y = [ex[-1:len(ex)][0] for ex in data]
            self.classes = self.get_classes(y)
            self.y = [self.classes[label] for label in y]
        self.num_ipregs = len(self.X[0])
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
        self.fitness = None
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
        p.effective_instrs = find_introns(v, p)
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
            data[i] = [float(x) for x in data[i]]
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

def split_data(X, y, standardize_method, alpha=1, validation=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, val0, val1 = standardize(X_train, standardize_method, alpha=alpha)
    X_test = standardize(X_test, standardize_method, alpha=alpha, vals=[val0, val1])[0]
    if validation:
        X_train, X_valid, y_train, y_valid, = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    else:
        return X_train, None, X_test, y_train, None, y_test

'''
Results
'''
def get_fitness_results(pop, X, y, v, fitness_eval, training=False):
    #results = [fitness_eval(prog, y, predicted_classes(prog, X, v)) if (prog.fitness is None or not training) else prog.fitness for prog in pop]
    processes = []
    if fitness_eval == fitness_sharing:
        results = fitness_sharing(pop, X, y, v, training)
    else:
        # q = mp.Queue()
        # missing_fitness_i = [i for i in range(len(pop)) if pop[i].fitness is None] if training else list(range(len(pop)))
        # if len(missing_fitness_i) > 1:
        #     procs = [Process(target=predicted_classes, args=(pop[i], X, v, i, q)) for i in missing_ypred_i]
        #     for proc in procs:
        #         p.start()
        #     for proc in procs:
        #         p.join()
        #     results = [fitness_eval(prog, y, predicted_classes(prog, X, v), training) if (prog.fitness is None or not training) else prog.fitness for prog in pop]
        #
        # # if len(missing_fitness_i) > 1:
        # #     with mp.Pool(2) as p:
        # #         results = [fitness_eval(prog, y, p.apply(predicted_classes, [prog, X, v]), training) if (prog.fitness is None or not training) else prog.fitness for prog in pop]
        #
        # else:
        #     results = [fitness_eval(prog, y, predicted_classes(prog, X, v), training) if (prog.fitness is None or not training) else prog.fitness for prog in pop]
        #
        #


        results = [fitness_eval(prog, y, predicted_classes(prog, X, v), training) if (prog.fitness is None or not training) else prog.fitness for prog in pop]
    return results



def predicted_classes(prog, X, v, i=None, q=None):
    y_pred = []
    for ex in X:
        output = v.run_prog(prog, ex)
        y_pred.append(output.index(max(output)))

    if q:
        q.put((y_pred, i))
    return y_pred


'''
Fitness evaluation functions
'''
def accuracy(prog, y, y_pred, training=False):
    acc = accuracy_score(y, y_pred)
    if training:
        prog.fitness = acc
    return acc

def avg_detect_rate(prog, y, y_pred, training=False):
    percentages = []
    for cl in  env.classes.values():
        cl_results = [i for i in range(len(y)) if y[i] == cl]
        percentages.append(sum([1 for i in cl_results if y[i] == y_pred[i]])/len(cl_results))
    fitness = np.mean(percentages)
    if training:
        prog.fitness = fitness
    return fitness

def fitness_sharing(pop, X, y, v, training=False):
    if training:
        for prog in pop:
            if prog.train_y_pred is None:
                prog.train_y_pred = predicted_classes(prog, X, v)
        all_y_pred = [prog.train_y_pred for prog in pop]
    else:
        all_y_pred = [predicted_classes(prog, X, v) for prog in pop]
    all_g = sum([sum([1 for i in range(len(y)) if y[i] == y_pred[i]]) for y_pred in all_y_pred])
    fitness = [sum([1 for i in range(len(y)) if y[i] == y_pred[i]])/(1+all_g) for y_pred in all_y_pred]
    return fitness


def ce(prog, y, y_pred):
    #ce =len([1 for i in range(len(y_pred)) if y_pred[i] != y[i]])/n
    #mse = sum([1 for i in range(len(y_pred)) if y_pred[i] != y[i]])*(1/n*3)
    ce = len([1 for i in range(len(y_pred)) if y_pred[i] != y[i]])
    return -(ce)

def cw_detection(prog, y, y_pred, classes):
    dr = []
    for cl in classes:
        tp = len([1 for i in range(len(y_pred)) if (y_pred[i] == cl) and (y_pred[i] == y[i])])
        fn = len([1 for i in range(len(y_pred)) if (y[i] == cl) and (y_pred[i] != y[i])])
    total = sum(dr)/len(classes)
    return total

'''
Variation operators
'''
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
        check_fitness_after_variation(prog, v, list(range(start_index, end_index)))
    return [progs[0], progs[1]]

# Mutation - change 1 value in the program
def mutation(progs, effective_mutations=True):
    step_size = 1
    progs = copy.deepcopy(progs)
    children = []
    for prog in progs:
        # Test - effective mutations
        if effective_mutations:
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
        #prog.effective_instrs, prog.fitness = None, None
        check_fitness_after_variation(prog, v, [index])
        children.append(prog)
    return children

'''
Selection
'''
# Steady state tournament for selection
def tournament(X, y, v, pop, fitness_eval=accuracy, var_op_probs=[0.5,0.5]):
    indivs = set()
    while len(indivs) < const.TOURNAMENT_SIZE:
        indivs.add(random.randint(0, len(pop)-1))
    selected_i = list(indivs)
    results = get_fitness_results([pop[i] for i in selected_i], X, y, v, fitness_eval=fitness_eval, training=True)

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
        #pop += mutation(copy.deepcopy(parents))
        progs = mutation(parents)
    elif var_op == 1:
        #pop += recombination(copy.deepcopy(parents))
        progs = recombination(parents)

    pop += progs
    return pop

# Breeder model for selection
def breeder(X, y, v, pop, fitness_eval=accuracy, gap=0.2, var_op_probs=[0.5,0.5]):
    env.pop_size = len(pop)
    results = get_fitness_results(pop, X, y, v, fitness_eval=fitness_eval, training=True)
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
            #new_pop += mutation(copy.deepcopy([pop[parents_i[0]]]))
            progs = mutation([pop[parents_i[0]]])
        elif var_op == 1:
            parents_i = np.random.choice(top_i, 2, replace=False).tolist()
            #new_pop += recombination(copy.deepcopy([pop[i] for i in parents_i]))
            progs = recombination([pop[i] for i in parents_i])

        new_pop += progs
    return new_pop

def check_fitness_after_variation(prog, v, instrs_changed):
    effective_instrs = v.find_introns(prog.prog[const.TARGET], prog.prog[const.SOURCE], prog.prog[const.MODE])
    if (effective_instrs == prog.effective_instrs) and set(instrs_changed).isdisjoint(effective_instrs):
        pass
    else:
        prog.fitness = None
        prog.train_y_pred = None
        prog.effective_instrs = effective_instrs

def run_model(X, y, v, pop, selection, generations, fitness_eval=accuracy, X_test=None, y_test=None):
    print_info()

    means = []
    top_results = []
    top_valid = []
    #print('Start fitness: ')
    #results = get_fitness_results(pop, X, y, v, fitness_eval, training=True)
    #print('{}:  max({}): {}\tmean:{}'.format(1, results.index(max(results)), max(results), np.mean(results)))
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
        pop = select(X, y, v, pop)

        results = get_fitness_results(pop, X, y, v, fitness_eval=fitness_eval, training=True)
        top_results.append(max(results))
        means.append(np.mean(results))

        if X_test and y_test:
            results = get_fitness_results([get_top_prog(pop, results)], X_test, y_test, v, fitness_eval=fitness_eval, training=False)
            top_valid.append(max(results))
    #graph(top_results, means, top_valid)
        # test_results = get_fitness_results([top_prog], X_test, y_test, v, fitness_eval)
        #print('Top on validation: {}'.format(test_results[0]))
    #print('\nEnd fitness: ')
    #results = get_fitness_results(pop, X, y, v, fitness_eval, training=True)
    #print('{}:  max({}): {}\tmean:{}'.format(env.generations, results.index(max(results)), max(results), np.mean(results)))
    #print(r)
    return pop

def graph(top_results, means, top_valid=None):
    generations = range(len(top_results))
    if top_valid:
        plt.plot(generations, top_results, generations, means, generations, top_valid)
    else:
        plt.plot(generations, top_results, generations, means)
    plt.show()

def get_average_fitness(X, y, v, num_trials, fitness_eval, validation=False):
    max_fitness , mean_fitness, test_fitness, valid_test_fitness, top_test_fitness_on_train = [], [], [], [], []
    top_test_fitness, valid_train_fitness = [], []
    test_accs, train_accs = [], []
    if validation:
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y, env.standardize_method, validation=True)
    else:
        X_train, temp, X_test, y_train, temp, y_test = split_data(X, y, env.standardize_method, validation=False)
    for i in range(num_trials):
        pop = gen_population(env.pop_size, env.num_ipregs)
        pop = run_model(X_train, y_train, v, pop, env.selection, env.generations, fitness_eval, X_test=X_test, y_test=y_test)
        results = get_fitness_results(pop, X_train, y_train, v, fitness_eval=fitness_eval, training=True)

        env.pop = pop
        env.X_train, env.X_test = X_train, X_test
        env.y_train, env.y_test = y_train, y_test

        max_fitness.append(max(results))
        mean_fitness.append(np.mean(results))

        top_prog = get_top_prog(pop, results)
        test_results = get_fitness_results([top_prog], X_test, y_test, v, fitness_eval)
        test_fitness.append(test_results[0])
        test_acc = get_fitness_results([top_prog], X_test, y_test, v, fitness_eval=accuracy)[0]
        train_acc = get_fitness_results([top_prog], X_train, y_train, v, fitness_eval=accuracy)[0]
        if fitness_eval != accuracy:
            test_accs.append(test_acc)
            train_accs.append(train_acc)
        print('Fitness on test data: {} (Acc: {})\nFitness on train data: {} (Acc: {})'.format(test_results[0], test_acc, max(results), train_acc))

        train_cl = class_percentages(top_prog, X_train, y_train, v, env.classes)
        test_cl = class_percentages(top_prog, X_test, y_test, v, env.classes)
        print('Test class percentages: {}\nTrain class percentages: {}'.format(test_cl, train_cl))

        all_test_results = get_fitness_results(pop, X_test, y_test, v, fitness_eval)
        top_test_result = max(all_test_results)
        #top_test_fitness.append(top_test_result)
        top_result_i = all_test_results.index(top_test_result)
        pop_score = get_fitness_results([pop[top_result_i]], X_train, y_train, v, fitness_eval)
        #top_test_fitness_on_train.append(pop_score[0])
        test_acc = get_fitness_results([pop[top_result_i]], X_test, y_test, v, fitness_eval=accuracy)[0]
        train_acc = get_fitness_results([pop[top_result_i]], X_train, y_train, v, fitness_eval=accuracy)[0]

        top_test_fitness.append(test_acc)
        top_test_fitness_on_train.append(train_acc)
        print('Top fitness on test data: {} (Acc: {})\nScore on train data for top: {} (Acc: {})'.format(top_test_result, test_acc, pop_score, train_acc))

        if validation:
            valid_fitness_eval = fitness_eval
            results = get_fitness_results(pop, X_valid, y_valid, v, fitness_eval=valid_fitness_eval)
            top_prog = get_top_prog(pop, results)
            test_results = get_fitness_results([top_prog], X_test, y_test, v, valid_fitness_eval)
            #valid_test_fitness.append(test_results[0])
            valid_score = get_fitness_results([top_prog], X_train, y_train, v, fitness_eval=fitness_eval)
            #valid_train_fitness.append(valid_score)

            test_acc = get_fitness_results([top_prog], X_test, y_test, v, fitness_eval=accuracy)[0]
            train_acc = get_fitness_results([top_prog], X_train, y_train, v, fitness_eval=accuracy)[0]
            valid_test_fitness.append(test_acc)
            valid_train_fitness.append(train_acc)
            print('Top fitness on validation data: {} (Acc: {}) \nScore on train data for top: {} (Acc: {})'.format(test_results[0], test_acc, valid_score, train_acc))
    avg_max, avg_mean, avg_test = np.mean(max_fitness), np.mean(mean_fitness), np.mean(test_fitness)
    avg_top_test, avg_top_on_train = np.mean(top_test_fitness), np.mean(top_test_fitness_on_train)
    avg_valid_test, avg_top_valid_on_train = np.mean(valid_test_fitness), np.mean(valid_train_fitness)
    print('Trials: {}\nAverage max fitness: {}\nAverage mean fitness: {}\nAverage test fitness: {}\n'.format(num_trials, avg_max, avg_mean, avg_test))
    print('Average top test acc: {}\nAverage top test program acc on training: {}\n'.format(avg_top_test, avg_top_on_train))
    if validation:
        print('Average test acc from top valid: {}\nAverage top valid program acc on training: {}\n'.format(avg_valid_test, avg_top_valid_on_train))
    if fitness_eval != accuracy:
        print('Average test accuracy: {}\nAverage train accuracy: {}\n'.format(np.mean(test_accs), np.mean(train_accs)))
def get_ranked_index(results):
    return [x[0] for x in sorted(enumerate(results), key=lambda i:i[1])]

def get_top_prog(pop, results):
    return pop[get_ranked_index(results)[-1]]

def run_top_prog(results, X_test, y_test, v, pop, fitness_eval=accuracy):
    top_prog = get_top_prog(pop, results)
    results0 = get_fitness_results([top_prog], X_test, y_test, v, fitness_eval)
    results1 = get_fitness_results([top_prog], X_train, y_train, v, fitness_eval)
    print('Fitness on test data: {}\nFitness on orig data: {}'.format(results0[0], results1[0]))
    all_results = get_fitness_results(pop, X_test, y_test, v, fitness_eval)
    top_result = max(all_results)
    top_result_i = all_results.index(max(all_results))
    pop_score = get_fitness_results([pop[top_result_i]], X_train, y_train, v, fitness_eval, training=True)
    print('Top fitness on test data: {} \nScore on orig data for top: {}'.format(top_result, pop_score))
    return top_prog


def find_introns(v, prog):
    return v.find_introns(prog.prog[const.TARGET], prog.prog[const.SOURCE], prog.prog[const.MODE])


def class_percentages(prog, X, y, v, classes):
    percentages = {}
    y_pred = predicted_classes(prog, X, v)

    for cl in classes:
        cl_results = [i for i in range(len(y)) if y[i] == classes[cl]]
        perc = sum([1 for i in cl_results if y[i] == y_pred[i]])/len(cl_results)
        percentages[cl] = perc
    return percentages

def print_info():
    print('Population size: {}\nGenerations: {}\nData: {}\nSelection Replacement: {}\nAlpha: {}\n'.format(env.pop_size, env.generations, env.data_file, env.selection.name, env.alpha))


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(env.X, env.y, env.standardize_method, validation=False)
    trials = 1
    get_average_fitness(env.X, env.y, v, trials, fitness_eval=env.fitness_eval)
env = Config()
env.load()
v = vm.Vm(const.GEN_REGS, env.num_ipregs, env.num_ops, env.output_dims)
#pop = gen_population(env.pop_size, env.num_ipregs)


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

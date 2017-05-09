import const, random, vm, copy, sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pdb

#input
num_genregs = 8
num_ops = 4
pop_size = 250
generations = 1000

data_files = ['data/iris.data', 'data/tic-tac-toe.data']
data_file = data_files[0]
standardize_method = const.StandardizeMethod.MEAN_VARIANCE
selection = const.Selection.STEADY_STATE_TOURN
alpha = 1


# Generating initial programs
def gen_prog(prog_length, num_ipregs):
    prog = [[],[],[],[]]
    for i in range(prog_length):
        prog[const.TARGET].append(random.randint(0, num_genregs-1))
        prog[const.SOURCE].append(random.randint(0, max(const.GEN_REGS, num_ipregs)-1))
        prog[const.OP].append(random.randint(0, num_ops-1))
        prog[const.MODE].append(random.randint(0, 1))
    return prog

def gen_population(pop_num, num_ipregs):
    return [gen_prog(const.PROG_LENGTH, num_ipregs) for n in range(0, pop_num)]

# Initializing the data
def load_data(fname):
    data = []
    with open(fname, 'r') as f:
        for line in f:
            data.append(line.split(','))
    return data

def standardize(data, method, alpha=10, vals=None):
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
        data[i] = [attrs.index(attr) for attr in data[i]]
    return data

def get_classes(data):
    classes = set(data)
    classmap = {}
    for i in range(len(classes)):
        classmap[classes.pop()] = i
    return classmap

# Results
def get_fitness_results(pop, X, y, v, fitness_eval):
    results = [fitness_eval(y, y_pred) for y_pred in [predicted_classes(prog, X, v) for prog in pop]]
    return results

def predicted_classes(prog, X, v):
    y_pred = []
    for ex in X:
        output = v.run_prog(prog, ex)
        y_pred.append(output.index(max(output)))
    return y_pred

# Fitness evaluation functions
def accuracy(y, y_pred):
    return accuracy_score(y, y_pred)

def mse_ce(y, y_pred):
    total = 0
    n = len(y)
    ce =len([1 for i in range(len(y_pred)) if y_pred[i] != y[i]])/n
    mse = sum([1 for i in range(len(y_pred)) if y_pred[i] != y[i]])*(1/n*3)
    #mse=0
    #ce = len([1 for i in range(len(y_pred)) if y_pred[i] != y[i]])
    return -(mse+ce)

# Variation operators
def reproduction(progs):
    return progs

def recombination(progs):
    assert len(progs) == 2
    start_index = random.randint(0, const.PROG_LENGTH-2)
    end_limit = const.PROG_LENGTH-1 if start_index is 0 else const.PROG_LENGTH
    end_index = random.randint(start_index+1, end_limit)    # avoid swapping whole program
    for col in range(len(progs[0])):
        progs[1][col][start_index:end_index], progs[0][col][start_index:end_index] = progs[0][col][start_index:end_index], progs[1][col][start_index:end_index]
    return [progs[0], progs[1]]

def mutation(progs):
    children = []
    for prog in progs:
        index = random.randint(0, const.PROG_LENGTH-1)
        col = random.randint(0, len(prog)-1)
        orig_val = prog[col][index]
        new_val = orig_val
        if col == const.OP:
            while new_val == orig_val:
                new_val = random.randint(0, max_vals[const.OP])
        else:
            op = random.randint(0, 1)
            while (new_val == orig_val):
                new_val = (orig_val+1) if op is 0 else (orig_val-1)
                if (new_val < 0) or (new_val > max_vals[col]):
                    new_val = orig_val
                    op = 0 if op is 1 else 1
        prog[col][index] = new_val
        children.append(prog)
    return children

def tournament(X, y, v, pop, fitness_eval=accuracy, var_op_probs=[0.1,0.9]):
    indivs = set()
    while len(indivs) < const.TOURNAMENT_SIZE:
        indivs.add(random.randint(0, len(pop)-1))
    selected_i = list(indivs)
    results = get_fitness_results([pop[i] for i in selected_i], X, y, v, fitness_eval=fitness_eval)

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
        pop += mutation(copy.deepcopy(parents))
    elif var_op == 1:
        pop += recombination(copy.deepcopy(parents))
    return pop

def breeder(X, y, v, pop, fitness_eval=accuracy, gap=0.2, var_op_probs=[0.5,0.5]):
    pop_size = len(pop)
    results = get_fitness_results(pop, X, y, v, fitness_eval=fitness_eval)
    ranked_index = get_ranked_index(results)

    partition = int(len(pop)*gap)
    top_i = ranked_index[-partition:]
    new_pop = [pop[i] for i in top_i]
    num_children = pop_size - len(new_pop)
    while len(new_pop) < pop_size:
        if len(new_pop) == (pop_size-1):
            var_op = 0
        else:
            var_op = np.random.choice([0,1], p=var_op_probs)

        if var_op == 0:
            parents_i = [np.random.choice(top_i)]
            new_pop += mutation(copy.deepcopy([pop[i] for i in parents_i]))
        elif var_op == 1:
            parents_i = np.random.choice(top_i, 2, replace=False).tolist()
            new_pop += recombination(copy.deepcopy([pop[i] for i in parents_i]))
    return new_pop

def run_model(X, y, v, pop, selection, generations, fitness_eval=accuracy):
    print('Start fitness: ')
    results = get_fitness_results(pop, X, y, v, fitness_eval)
    print('{}:  max({}): {}\tmean:{}'.format(1, results.index(max(results)), max(results), np.mean(results)))
    if selection == const.Selection.STEADY_STATE_TOURN:
        select = tournament
    elif selection == const.Selection.BREEDER_MODEL:
        select = breeder
    else:
        raise ValueError('Invalid selection: {}'.format(selection))

    for i in range(generations):
        print(i, end=' ')
        sys.stdout.flush()
        assert len(pop) == pop_size
        pop = select(X, y, v, pop)
    print('\nEnd fitness: ')
    results = get_fitness_results(pop, X, y, v, fitness_eval)
    print('{}:  max({}): {}\tmean:{}'.format(generations, results.index(max(results)), max(results), np.mean(results)))
    #print(r)
    return results

def get_average_fitness(X, y, v, pop, selection, generations, num_trials, fitness_eval=accuracy):
    max_fitness , mean_fitness, test_fitness = [], [], []
    X_train, X_test, y_train, y_test = X[0], X[1], y[0], y[1]
    for i in range(num_trials):
        results = run_model(X_train, y_train, v, pop, selection, generations, fitness_eval)
        max_fitness.append(max(results))
        mean_fitness.append(np.mean(results))
        test_fitness.append(run_top_prog(results, X_test, y_test, v, pop))
        pop = gen_population(pop_size, num_ipregs)
    avg_max, avg_mean, avg_test = np.mean(max_fitness), np.mean(mean_fitness), np.mean(test_fitness)
    print('Trials: {}\nAverage max fitness: {}\nAverage mean fitness: {}\nAverage test fitness: {}\n'.format(num_trials, avg_max, avg_mean, avg_test))
    return avg_max, avg_mean, avg_test

def get_ranked_index(results):
    return [x[0] for x in sorted(enumerate(results), key=lambda i:i[1])]

def run_top_prog(results, X, y, v, pop, fitness_eval=accuracy):
    top_prog = pop[get_ranked_index(results)[-1]]
    results = get_fitness_results([top_prog], X, y, v, fitness_eval)
    print('Fitness on test data: {}'.format(results[0]))
    all_results = get_fitness_results(pop, X, y, v, fitness_eval)
    top_result = max(all_results)
    top_result_i = all_results.index(max(all_results))
    pop_score= get_fitness_results([pop[top_result_i]], X_train, y_train, v, fitness_eval)
    print('Top fitness on test data: {} \nScore on orig data: {}'.format(top_result, pop_score))
    return results[0]

var_ops = [recombination, mutation, reproduction]
fitness_eval = accuracy

data = load_data(data_file)
X, y = preprocess([ex[:len(ex)-1] for ex in data]), [ex[-1:len(ex)][0] for ex in data]
classes = get_classes(y)
y = [classes[label] for label in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
output_dims = len(classes)
num_ipregs = len(X[0])
pop = gen_population(pop_size, num_ipregs)
max_vals = [num_genregs-1, max(num_genregs, num_ipregs)-1, num_ops-1, 1]
v = vm.Vm(num_genregs, num_ipregs, num_ops, output_dims)

X_train, val0, val1 = standardize(X_train, standardize_method, alpha=alpha)
X_test = standardize(X_test, standardize_method, alpha=alpha, vals=[val0, val1])[0]

X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.2, stratify=y_test)
#run_model(X_train, y_train, v, pop, selection, generations)
# results = get_fitness_results(pop, X_valid, y_valid, v, fitness_eval=fitness_eval)
# run_top_prog(results, X_test, y_test, v, pop)


print('Population size: {}\nGenerations: {}\nData: {}\nSelection Replacement: {}\nAlpha: {}\n'.format(pop_size, generations, data_file, selection.name, alpha))
#results = run_model(X_train, y_train, v, pop, selection, generations)

trials = 2
get_average_fitness([X_train, X_test], [y_train, y_test], v, pop, selection, generations, trials, fitness_eval=fitness_eval)

#run_top_prog(results, X_test, y_test, v, pop)

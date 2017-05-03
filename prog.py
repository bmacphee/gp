import const, random, vm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#input
num_genregs = 8
num_ipregs = 4
num_ops = 4
output_dims = 3
data_file = 'data/iris.data'

def gen_prog(prog_length):
    prog = [[],[],[],[]]
    for i in range(0, prog_length):
        prog[const.TARGET].append(random.randint(0, num_genregs-1))
        prog[const.SOURCE].append(random.randint(0, max(num_genregs, num_ipregs)-1))
        prog[const.OP].append(random.randint(0, num_ops-1))
        prog[const.MODE].append(random.randint(0, 1))
    return prog

def gen_population(pop_num):
    return [gen_prog(const.PROG_LENGTH) for n in range(0, pop_num)]

def load_data(fname):
    data = []
    with open(fname, 'r') as f:
        for line in f:
            data.append(line.split(','))
    return preprocess(data)

def preprocess(data):
    return data

def get_classes(data):
    classes = set(data)
    classmap = {}
    for i in range(0, len(classes)):
        classmap[classes.pop()] = i
    return classmap

data = load_data(data_file)
X, y = [ex[:len(ex)-1] for ex in data], [ex[-1:len(ex)][0] for ex in data]
classes = get_classes(y)
y = [classes[label] for label in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
pop = gen_population(2)
v = vm.Vm(num_genregs, num_ipregs, num_ops, output_dims)


def get_results(pop):
    results=[]
    for indiv in pop:
        y_pred = []
        for i, ex in enumerate(X_train):
            output = v.run_prog(indiv, ex)
            y_pred.append(output.index(max(output)))
        print(y_pred)
        results.append(accuracy_score(y_train, y_pred))
    return results

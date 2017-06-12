import vm, pdb
import numpy as np
from sklearn.metrics import accuracy_score
from array import array
import const

'''
Fitness evaluation functions
'''


def accuracy(prog, y, y_pred, store_fitness=None):
    acc = accuracy_score(y, y_pred)
    if store_fitness:
        setattr(prog, store_fitness, acc)
    return acc


# @profile
def avg_detect_rate(prog, y, y_pred, store_fitness=None):
    # percentages = []
    # for cl in set(y):
    #     cl_results = [i for i in range(len(y)) if y[i] == cl]
    #     percentages.append(sum([1 for i in cl_results if y[i] == y_pred[i]]) / len(cl_results))
    # fitness = np.mean(percentages)
    # pdb.set_trace()
    fitness = vm.avg_detect_rate(prog, y, array('i', y_pred))
    return fitness


# @profile
def fitness_sharing(pop, X, y):
    # training = (store_fitness == 'trainset_trainfit')
    # all_y_pred = [predicted_classes(prog, X, fitness_sharing=training) for prog in pop]
    # all_y_pred = vm.y_pred(np.array(pop), np.array(X), 0)

    # all_y_pred = vm.y_pred(pop, X)
    # denoms = [sum([1 for j in range(len(all_y_pred)) if all_y_pred[j][i] == y[i]]) + 1 for i in range(len(y))]
    # fitness = [sum([int(y_pred[i] == y[i]) / denoms[i] for i in range(len(y))]) for y_pred in all_y_pred]
    # return fitness
    fitness = vm.fitness_sharing(pop, X, y)
    return fitness


# @profile
def predicted_classes(prog, X, fitness_sharing=0):
    # set_train_pred = not env.use_subset and (fitness_sharing)

    # if set_train_pred and (prog.train_y_pred[0] != -1):
    #     return prog.train_y_pred

    # y_pred = array('i', [-1]*len(X))
    y_pred = []
    for i in range(len(X)):
        # y_pred[i] = vm.run_prog(prog, X[i])
        y_pred.append(vm.run_prog(prog, X[i]))
    # if set_train_pred:
    #     prog.train_y_pred = y_pred
    return y_pred


def class_percentages(prog, X, y, classes):
    percentages = {}
    y_pred = predicted_classes(prog, X)

    for cl in classes:
        cl_results = [i for i in range(len(y)) if y[i] == classes[cl]]
        perc = sum([1 for i in cl_results if y[i] == y_pred[i]]) / len(cl_results)
        percentages[cl] = perc
    return percentages


# @profile
def find_introns(prog):
    instrs = vm.find_introns(prog.prog[const.TARGET], prog.prog[const.SOURCE], prog.prog[const.MODE])
    prog.effective_instrs = instrs
    return instrs

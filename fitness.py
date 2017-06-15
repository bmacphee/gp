import const, pdb
import numpy as np
import cythondir.vm as vm
from sklearn.metrics import accuracy_score
from array import array

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
    fitness = vm.avg_detect_rate(prog, y, array('i', y_pred))
    return fitness


#@profile
def fitness_sharing(pop, X, y):
    fitness = vm.fitness_sharing(pop, X, y)
    return fitness

#@profile
def predicted_classes(prog, X, fitness_sharing=0):
    y_pred = []
    for i in range(len(X)):
        y_pred.append(prog.run_prog(X[i]))
    return y_pred


def class_percentages(prog, X, y, classes):
    percentages = {}
    y_pred = vm.y_pred(np.asarray([prog]), X)[0]

    for cl in classes:
        cl_results = [i for i in range(len(y)) if y[i] == classes[cl]]
        perc = sum([1 for i in cl_results if y[i] == y_pred[i]]) / len(cl_results)
        percentages[cl] = perc
    return percentages


def find_introns(prog):
    instrs = vm.find_introns(prog.prog[const.TARGET], prog.prog[const.SOURCE], prog.prog[const.MODE])
    prog.effective_instrs = instrs
    return instrs

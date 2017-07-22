import const, pdb
import numpy as np
import cythondir.vm as vm
from sklearn.metrics import accuracy_score
from array import array

'''
Fitness evaluation functions
'''


def accuracy(prog, y, y_pred):
    acc = accuracy_score(y, y_pred)
    return acc


#@profile
def avg_detect_rate(prog, y, y_pred):
    fitness = vm.avg_detect_rate(prog, y, array('i', y_pred))
    return fitness


#@profile
def fitness_sharing(pop, X, y, hosts=None, curr_i=None):
    fitness = vm.fitness_sharing(pop, X, y, hosts, curr_i)
    return fitness

#@profile
def predicted_classes(prog, X, fitness_sharing=0):
    y_pred = []
    for i in range(len(X)):
        prog.run_prog(X[i])
        y_pred.append(prog.max_ind())
    return y_pred


def class_percentages(prog, X, y, classes, host=None):
    percentages = {}

    if host is not None:
        y_pred = vm.host_y_pred(np.asarray(prog), np.asarray([host]), X, None, 0)[0]
    else:
        y_pred = vm.y_pred(np.asarray([prog]), X)[0]

    for cl in classes:
        cl_results = [i for i in range(len(y)) if y[i] == classes[cl]]
        perc = sum([1 for i in cl_results if y[i] == y_pred[i]]) / len(cl_results)
        percentages[cl] = perc
    return percentages


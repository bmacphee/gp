import const, pdb
import numpy as np
import utils, cythondir.vm as vm
from sklearn.metrics import accuracy_score
from array import array

'''
Fitness evaluation functions
'''

def fitness_results(pop, X, y_act, fitness_eval, hosts=None, curr_i=None):
    pop_arr = np.asarray(pop)
    if fitness_eval.__name__ == 'fitness_sharing':
        results = fitness_sharing(pop_arr, X, y_act, hosts, curr_i)
    else:
        all_y_pred = vm.y_pred(pop_arr, X) if hosts is None else vm.host_y_pred(pop_arr, hosts, X, None, 0)
        results = [fitness_eval(pop[i], y_act, all_y_pred[i]) for i in range(len(all_y_pred))]
    return results


def accuracy(prog, y, y_pred, hosts=None):
    acc = accuracy_score(y.tolist(), y_pred)
    return acc


#@profile
def avg_detect_rate(prog, y_act, y_pred):
    y_act = array('i', y_act) if type(y_act) == list else y_act
    fitness = vm.avg_detect_rate(prog, y_act, array('i', y_pred))
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


def cumulative_detect_rate(data, pop, hosts, testset_with_testfit, trainset_with_testfit):
    # Move this later - is calculated twice
    if hosts is not None:
        curr_hosts = utils.get_nonzero(hosts)
        y_pred = vm.host_y_pred(np.asarray(pop), curr_hosts, data.X_test, None, 0)
    else:
        y_pred = vm.y_pred(np.asarray(pop), data.X_test).base
    detect_rates = []
    ranked = utils.get_ranked_index(trainset_with_testfit)
    top = y_pred[ranked[-1]]

    while len(ranked) > 0:
        addition = y_pred[ranked.pop()]
        top[addition == data.y_test] = addition[addition == data.y_test]
        detect_rate = vm.avg_detect_rate(None, data.y_test, array('i', top))
        detect_rates.append(detect_rate)
        if detect_rate == 1:
            break
    detect_rates += [1.0] * (len(ranked))
    return detect_rates

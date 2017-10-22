import pdb
import numpy as np
import utils, cythondir.vm as vm
from sklearn.metrics import accuracy_score
from array import array

'''
Fitness evaluation functions
'''


# def fitness_results(traintest, system, X, y, fitness_eval, data_i=None, hosts_i=None):
#     if fitness_eval.__name__ == 'fitness_sharing':
#         results = fitness_sharing(system, X, y, data_i, hosts_i)
#     else:
#         all_y_pred = system.y_pred(X, traintest=traintest, hosts_i=hosts_i, data_i=data_i, training=training)
#         results = [fitness_eval(y, all_y_pred[i]) for i in range(len(all_y_pred))]
#     return results


def accuracy(prog, y, y_pred, hosts=None):
    acc = accuracy_score(y.tolist(), y_pred)
    return acc


# @profile
def avg_detect_rate(y_act, y_pred):
    y_act = array('i', y_act) if type(y_act) == list else y_act
    fitness = vm.avg_detect_rate(y_act, array('i', y_pred))
    return fitness


#@profile
def fitness_sharing(system, X, y, data_i=None, hosts_i=None):
    if hosts_i is None and system.hosts is not None:
        hosts_i = range(len(system.hosts))
    fitness = vm.fitness_sharing(system.pop, X, y, system.hosts, data_i, array('i', hosts_i))
    return fitness


# @profile
def predicted_classes(prog, X, fitness_sharing=0):
    y_pred = []
    for i in range(len(X)):
        prog.run_prog(X[i])
        y_pred.append(prog.max_ind())
    return y_pred


def class_percentages(system, X, y, classes, train_test, hosts_i=None, data_i=None):
    percentages = {}
    y_pred = system.y_pred(X, traintest=train_test, hosts_i=hosts_i, data_i=data_i)[0]
    for cl in classes:
        cl_results = [i for i in range(len(y)) if y[i] == classes[cl]]
        if cl_results:
            perc = sum([1 for i in cl_results if y[i] == y_pred[i]]) / len(cl_results)
        else:
            perc = 0
        percentages[cl] = perc
    return percentages


def cumulative_detect_rate(data, pop, hosts, trainset_with_testfit, hosts_i=None):
    # Move this later - is calculated twice
    try:
        if hosts is not None:
            if hosts_i is None:
                curr_hosts = utils.get_nonzero(hosts)
                hosts_i = array('i', range(len(curr_hosts)))  ## TODO: check this
            else:
                curr_hosts = hosts
            data_i = array('i', range(len(data.X_test)))
            y_pred = vm.host_y_pred(pop, curr_hosts, data.X_test, data_i, 1, 0, array('i', hosts_i)).base
        else:
            y_pred = vm.y_pred(pop, data.X_test).base
        ranked = utils.get_ranked_index(trainset_with_testfit)
        top = y_pred[ranked[-1]]

        detect_rates = []
        while len(ranked) > 0:
            addition = y_pred[ranked.pop()]
            top[addition == data.y_test] = addition[addition == data.y_test]
            detect_rate = vm.avg_detect_rate(data.y_test, array('i', top))
            detect_rates.append(detect_rate)
            if detect_rate == 1:
                break
        detect_rates += [1.0] * (len(ranked))

    except:
        pdb.set_trace()
    return detect_rates

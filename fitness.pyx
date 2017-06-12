import vm, pdb, const
import numpy as np
cimport numpy as np
cimport cython
from sklearn.metrics import accuracy_score
from array import array

from libc.stdlib cimport malloc, free
from libc.stdio cimport printf


'''
Fitness evaluation functions
'''
cpdef accuracy(prog, y, y_pred, store_fitness=None):
    acc = accuracy_score(y, y_pred)
    if store_fitness:
        setattr(prog, store_fitness, acc)
    return acc


#@profile
cpdef list avg_detect_rate(prog, y, y_pred, store_fitness=None):
    cpdef:
        list percentages = []
        int y, i
    for cl in set(y):
        cl_results = [i for i in range(len(y)) if y[i] == cl]
        percentages.append(sum([1 for i in cl_results if y[i] == y_pred[i]])/len(cl_results))
    fitness = np.mean(percentages)
    # if store_fitness:
    #     setattr(prog, store_fitness, fitness)
    return fitness

#@profile
cpdef fitness_sharing(pop, X, y):
    #training = (store_fitness == 'trainset_trainfit')
    #all_y_pred = [predicted_classes(prog, X, fitness_sharing=training) for prog in pop]
    #all_y_pred = vm.y_pred(np.array(pop), np.array(X), 0)

    # all_y_pred = vm.y_pred(pop, X)
    # denoms = [sum([1 for j in range(len(all_y_pred)) if all_y_pred[j][i] == y[i]]) + 1 for i in range(len(y))]
    # fitness = [sum([int(y_pred[i] == y[i]) / denoms[i] for i in range(len(y))]) for y_pred in all_y_pred]
    # return fitness
    fitness = vm.fitness_sharing(pop, X, y)
    return fitness

#@profile
cpdef predicted_classes(prog, X, fitness_sharing=0):
    #set_train_pred = not env.use_subset and (fitness_sharing)

    # if set_train_pred and (prog.train_y_pred[0] != -1):
    #     return prog.train_y_pred

    #y_pred = array('i', [-1]*len(X))
    y_pred = []
    for i in range(len(X)):
        #y_pred[i] = vm.run_prog(prog, X[i])
        y_pred.append(vm.run_prog(prog, X[i]))
    # if set_train_pred:
    #     prog.train_y_pred = y_pred
    return y_pred

cpdef dict class_percentages(prog, X, y, classes):
    percentages = {}
    y_pred = predicted_classes(prog, X)

    for cl in classes:
        cl_results = [i for i in range(len(y)) if y[i] == classes[cl]]
        perc = sum([1 for i in cl_results if y[i] == y_pred[i]])/len(cl_results)
        percentages[cl] = perc
    return percentages

#@profile
cpdef array.array[int] find_introns(vm.Prog prog):
    instrs = vm.find_introns(prog.prog[const.TARGET], prog.prog[const.SOURCE], prog.prog[const.MODE])
    prog.effective_instrs = instrs
    return instrs

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list y_pred(list progs, np.ndarray X):
    cdef int num_progs = len(progs), num_ex = len(X)
    cdef np.ndarray all_y_pred = np.empty((num_progs, num_ex), dtype=np.int32_t)
    cdef array.array[int] y_pred = array.array('i', range(num_ex))
    cdef int i, j
    cdef Prog prog

    for i in range(num_progs):
        prog = progs[i]
        for j in range(num_ex):
            y_pred[j] = run_prog(prog, X[j])
        all_y_pred[i] = y_pred
    return all_y_pred.tolist()


cpdef list fitness_sharing(list pop, np.ndarray X, array.array[int] y):
    cdef:
        list all_y_pred
        list denoms = []
        list fitness = []
        int i, j, len_y = len(y), len_pop = len(pop), sum
    all_y_pred = y_pred(pop, X)

    for i in range(len_y):
        sum = 0
        for j in range (len_pop):
            if all_y_pred[j][i] == y[i]:
                sum += 1
        denoms.append(sum + 1)

    for i in range(len_pop):
        sum = 0
        for j in range(len_y):
            if all_y_pred[i][j] == y[j]:
                sum += 1
        fitness.append(sum/denoms[i])
    #
    # denoms = [sum([1 for j in range(len(all_y_pred)) if all_y_pred[j][i] == y[i]]) + 1 for i in range(len(y))]
    # fitness = [sum([int(y_pred[i] == y[i]) / denoms[i] for i in range(len(y))]) for y_pred in all_y_pred]
    return fitness
import numpy as np, pdb, sys
import utils
from cythondir.vm import host_y_pred, y_pred
from array import array
import fitness as fit


class System:
    hosts = None

    def __init__(self, pop, hosts):
        System.hosts = hosts
        self.pop = pop
        self.root_hosts = lambda: None
        self.curr_hosts = lambda: None
        self.tangled_graphs = 0

    def y_pred(self, X, traintest=None, hosts_i=None, data_i=None):
        if hosts_i:
            pop = [self.pop[i] for i in hosts_i]
        return y_pred(pop, X)

    def trainset_fitness_results(self, X, y, fitness_eval, data_i=[]):
        return fit.fitness_results(0, self, X, y, fitness_eval, data_i=array('i', data_i), hosts_i=range(len(self.pop)))

    def testset_fitness_results(self, X, y, fitness_eval, results):
        data_i = array('i', range(len(X)))
        pop_i = [utils.get_ranked_index(results)[-1]]
        return fit.fitness_results(1, self, X, y, fitness_eval, data_i=array('i', data_i), hosts_i=pop_i)


class BidSystem(System):
    def __init__(self, pop, hosts):
        super().__init__(pop, hosts)
        self.curr_hosts = lambda: self.hosts[np.nonzero(self.hosts)[0]]
        self.root_hosts = lambda: np.nonzero(self.hosts)[0]

    def y_pred(self, X, traintest=None, hosts_i=None, data_i=None):
        if hosts_i is None:
            hosts_i = self.root_hosts()
        try:
            return host_y_pred(self.pop, self.hosts, X, data_i, traintest, 0, array('i', hosts_i))
        except:
            pdb.set_trace()

    def trainset_fitness_results(self, X, y, fitness_eval, data_i=None):
        return fit.fitness_results(0, self, X, y, fitness_eval, data_i, self.root_hosts())

    def testset_fitness_results(self, X, y, fitness_eval, results):
        data_i = array('i', range(len(X)))
        hosts_i = [self.root_hosts()[utils.get_ranked_index(results)[-1]]]
        fitness = fit.fitness_results(1, self, X, y, fitness_eval, data_i, hosts_i)
        try:
            assert fitness == fit.fitness_results(1, self, X, y, fitness_eval, None, hosts_i)  # Testing
        except:
            pdb.set_trace()
        return fitness


class GraphSystem(BidSystem):
    def __init__(self, pop, hosts):
        super().__init__(pop, hosts)
        self.tangled_graphs = 1
        self.root_hosts = lambda: [h.index_num for h in self.curr_hosts() if h.num_refs == 0]


def print_graphs(system, input, ind=None):
    assert isinstance(system, GraphSystem), 'Cannot print graphs for non-graph system'
    if ind is None:
        ind = system.root_hosts()
    if type(ind) != list:
        ind = list(ind)

    for i in ind:
        _print_path(system, input, i)
    print('\n')
    sys.stdout.flush()


def _print_path(system, input, i):
    print('Host {} -->'.format(i), end=' ')
    vals = []
    for prog_i in system.hosts[i].progs_i:
        prog = system.pop[prog_i]
        prog.run_prog(input)
        vals.append(prog.get_regs()[0])
    winner_i = system.hosts[i].progs_i[vals.index(max(vals))]
    winner = system.pop[winner_i]
    print('Prog: {} -->'.format(winner_i), end=' ')
    if winner.atomic_action == 1:
        print('Class {}'.format(winner.class_label))
    elif winner.atomic_action == 0:
        _print_path(system, input, winner.class_label)

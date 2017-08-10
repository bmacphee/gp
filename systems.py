import numpy as np, pdb
import utils
from cythondir.vm import host_y_pred, y_pred
from array import array
import fitness as fit
#
class System:
    hosts = None

    def __init__(self, pop, hosts, tangled_graphs):
        self.pop = pop
        System.hosts = hosts
        self.root_hosts = lambda: None
        self.curr_hosts = lambda: None

    def y_pred(self, X, traintest=None, select_i=None, data_i=None):
        if select_i:
            pop = [self.pop[i] for i in select_i]
        return y_pred(pop, X)

    def trainset_fitness_results(self, X, y, fitness_eval, data_i=[]):
        return fit.fitness_results(0, self, X, y, fitness_eval, data_i=array('i', data_i), select_i=range(len(self.pop)))

    def testset_fitness_results(self, X, y, fitness_eval, results, data_i=[]):
        pop_i = [utils.get_ranked_index(results)[-1]]
        return fit.fitness_results(1, self, X, y, fitness_eval, data_i=array('i', data_i), select_i=pop_i)


class BidSystem(System):
    def __init__(self, pop, hosts, tangled_graphs):
        super().__init__(pop, hosts, tangled_graphs)
        self.curr_hosts = lambda: self.hosts[np.nonzero(self.hosts)[0]]
        self.root_hosts = lambda: np.nonzero(self.hosts)[0]

    def y_pred(self, X, traintest=None, select_i=None, data_i=None):
        if select_i is None:
            select_i = self.root_hosts()
        try:
            return host_y_pred(self.pop, self.hosts, X, data_i, traintest, 0, array('i', select_i))
        except:
            pdb.set_trace()

    def trainset_fitness_results(self, X, y, fitness_eval, data_i=None):
        return fit.fitness_results(0, self, X, y, fitness_eval, data_i, self.root_hosts())

    def testset_fitness_results(self, X, y, fitness_eval, results):
        data_i = range(len(X))
        select_i = [self.root_hosts()[utils.get_ranked_index(results)[-1]]]
        return fit.fitness_results(1, self, X, y, fitness_eval, data_i, select_i)


class GraphSystem(BidSystem):
    def __init__(self, pop, hosts, tangled_graphs):
        super().__init__(pop, hosts, tangled_graphs)
        self.root_hosts = lambda: [h.index_num for h in self.curr_hosts() if h.num_refs == 0]
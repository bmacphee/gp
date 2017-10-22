import numpy as np, pdb, sys
import utils, const, os
from cythondir.vm import host_y_pred, y_pred
from array import array
import fitness as fit
import matplotlib

matplotlib.use("Agg")
from PIL import Image


class System:
    hosts = None

    def __init__(self, pop, hosts=None):
        System.hosts = hosts
        self.pop = pop
        self.tangled_graphs = 0

    @property
    def root_hosts(self):
        return None

    @property
    def curr_hosts(self):
        return None

    def y_pred(self, X, traintest=None, hosts_i=None, data_i=None):
        if hosts_i:
            pop = np.array([self.pop[i] for i in hosts_i])
        return y_pred(pop, X)

    def trainset_fitness_results(self, X, y, fitness_eval, data_i=list(), validation=0):
        return self.fitness_results(0, X, y, fitness_eval, data_i=array('i', data_i),
                                    hosts_i=array('i', list(range(len(self.pop)))), validation=validation)

    def testset_fitness_results(self, X, y, fitness_eval, results):
        data_i = array('i', range(len(X)))
        pop_i = array('i', [utils.get_ranked_index(results)[-1]])
        return self.fitness_results(1, X, y, fitness_eval, data_i=array('i', data_i), hosts_i=pop_i)

    def fitness_results(self, traintest, X, y, fitness_eval, data_i=None, hosts_i=None, validation=0):
        if fitness_eval.__name__ == 'fitness_sharing':
            results = fit.fitness_sharing(self, X, y, data_i, hosts_i)
        else:
            all_y_pred = self.y_pred(X, traintest=traintest, hosts_i=hosts_i, data_i=data_i, validation=validation)
            results = [fitness_eval(y, all_y_pred[i]) for i in range(len(all_y_pred))]
        return results


class BidSystem(System):
    def __init__(self, pop, hosts):
        super().__init__(pop, hosts)

    @property
    def curr_hosts(self):
        return self.hosts[np.nonzero(self.hosts)[0]]

    @property
    def root_hosts(self):
        return np.nonzero(self.hosts)[0]

    def y_pred(self, X, traintest=None, hosts_i=None, data_i=None, validation=0):
        if hosts_i is None:
            hosts_i = self.root_hosts

        if validation:
            change_regs = 0
        else:
            change_regs = 1 - traintest
        return host_y_pred(self.pop, self.hosts, X, data_i, traintest, change_regs, array('i', hosts_i))

    def trainset_fitness_results(self, X, y, fitness_eval, data_i=None, validation=0):
        return self.fitness_results(0, X, y, fitness_eval, data_i=data_i, hosts_i=self.root_hosts,
                                    validation=validation)

    def testset_fitness_results(self, X, y, fitness_eval, ind):
        data_i = array('i', range(len(X)))
        hosts_i = np.array([ind])
        fitness = self.fitness_results(1, X, y, fitness_eval, data_i, hosts_i)
        try:
            assert fitness == self.fitness_results(1, X, y, fitness_eval, None, hosts_i)  # Testing
        except:
            pdb.set_trace()
        return fitness


class GraphSystem(BidSystem):
    def __init__(self, pop, hosts):
        super().__init__(pop, hosts)
        self.tangled_graphs = 1

    @property
    def root_hosts(self):
        return array('i', [h.index_num for h in self.curr_hosts if h.num_refs == 0])


def print_graphs(system, input, ind=None):
    assert isinstance(system, GraphSystem), 'Cannot print graphs for non-graph system'
    if ind is None:
        ind = system.root_hosts
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


def print_top_host(nodes, edges, path, data, filenum):
    label_file = '{}/legal-actions.rslt'.format(const.GRAPHINFO_DIR)
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    with open(label_file, 'w') as f:
        f.write(' '.join(sorted([str(x) for x in data.classes.keys()])))
        f.write('\n')

    to_write = [('Links-host', 'from', 'to', edges), ('Nodes-host', 'id', 'type', nodes)]
    for item in to_write:
        filename = '{}_{}'.format(item[0], filenum)
        filepath = os.path.join(const.GRAPHINFO_DIR, filename)
        with open(filepath, 'w') as f:
            f.write('{},{}\n'.format(item[1], item[2]))
            for pair in item[3]:
                f.write('{},{}\n'.format(pair[0], pair[1]))

    if path:
        path_file = '{}/active-nodes.rslt'.format(os.path.dirname(const.GRAPHINFO_DIR))
        with open(path_file, 'w') as f:
            f.write(' '.join([str(x[0]) for x in path]))
            f.write('\n')

# def print_top_host(system, env, data, stats):
#     assert type(system) is GraphSystem, 'GraphSystem required'
#     top = utils.top_host_i(stats, system)
#     print_nodes_host(system, top, data, env.file_prefix)
#
#
# def print_nodes_host(system, host_i, data, filenum, path_i=None):
#     nodes = set()
#     edges = set()
#     path = set()
#     node = host_n(host_i)
#     if path_i is not None:
#         path.add(node)
#     nodes.add(node)
#     label_file = '{}/legal-actions.rslt'.format(const.GRAPHINFO_DIR)
#     os.makedirs(os.path.dirname(label_file), exist_ok=True)
#     with open(label_file, 'w') as f:
#         f.write(' '.join(sorted([str(x) for x in data.classes.keys()])))
#         f.write('\n')
#
#     for i, ex in enumerate(data.X_test):
#         get_traversed(system, host_i, ex, nodes, edges, node, i is path_i, path)
#
#     to_write = [('Links-host', 'from', 'to', edges), ('Nodes-host', 'id', 'type', nodes)]
#     for item in to_write:
#         filename = '{}_{}'.format(item[0], filenum)
#         filepath = os.path.join(const.GRAPHINFO_DIR, filename)
#         with open(filepath, 'w') as f:
#             f.write('{},{}\n'.format(item[1], item[2]))
#             for pair in item[3]:
#                 f.write('{},{}\n'.format(pair[0], pair[1]))
#
#     if path:
#         path_file = '{}/active-nodes.rslt'.format(os.path.dirname(const.GRAPHINFO_DIR))
#         with open(path_file, 'w') as f:
#             f.write(' '.join([str(x[0]) for x in path]))
#             f.write('\n')
#
#     return nodes, edges, path
#
#
# def get_traversed(system, host_i, X, nodes, edges, last_node, on_path, path=None):
#     vals = []
#     for prog_i in system.hosts[host_i].progs_i:
#         prog = system.pop[prog_i]
#         prog.run_prog(X)
#         vals.append(prog.get_regs()[0])
#     winner_i = system.hosts[host_i].progs_i[vals.index(max(vals))]
#     winner = system.pop[winner_i]
#
#     node0 = prog_n(winner.prog_id)
#     nodes.add(node0)
#     edges.add((last_node[0], node0[0]))
#     if on_path:
#         path.add(node0)
#
#     if winner.atomic_action == 1:
#         node1 = action_n(winner.prog_id, winner.class_label)
#     else:
#         node1 = host_n(winner.class_label)
#         if on_path:
#             path.add(node1)
#     nodes.add(node1)
#     edges.add((node0[0], node1[0]))
#     if winner.atomic_action == 0:
#         get_traversed(system, winner.class_label, X, nodes, edges, node1, on_path, path)
#
#
# def host_n(id):
#     return ('h{}'.format(id), 'host')
#
#
# def prog_n(id):
#     return ('s{}'.format(id), 'symbiont')
#
#
# def action_n(id, class_label):
#     return ('a{}'.format(id), 'atomic-{}'.format(class_label))


def get_indexed(prog, dims):
    pixels = np.zeros(dims[0] * dims[1], dtype=int)
    assert prog is not None, 'Prog does not exist'
    mode_col = prog.prog[const.MODE]
    source_col = prog.prog[const.SOURCE]
    rows = [i for i in range(len(mode_col)) if mode_col[i] == const.IP_MODE_VAL]
    ip_vals = [source_col[i] for i in rows]
    pixels[ip_vals] = 1
    return pixels.reshape(dims[0], dims[1])


def print_indexed(X, syst, prog, dims=(28, 28)):
    indexed = get_indexed(prog, dims)
    orig = np.array(X, dtype=np.uint8).reshape((28, 28))
    im = Image.fromarray(orig).convert('RGB')
    nonzero = np.nonzero(indexed)
    for x, y in zip(nonzero[0], nonzero[1]):
        im.putpixel((x, y), (255, 0, 0))
    im.save('{}/{}_{}_indexed.png'.format(const.INDEX_DIR, prog.prog_id, prog.grid_section))


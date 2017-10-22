import const, data_utils as dutil, fitness as fit
import time
from array import array


class Config(object):
    def __init__(self):
        self.random_seed = 1
        self.data_files = {
            0: 'data/iris.data',
            1: 'data/tic-tac-toe.data',
            2: 'data/ann-train.data',
            3: 'data/shuttle.trn',
            4: 'data/MNIST',
            5: 'load_mnist',
            6: 'data/SVHN/svhn_train_grayscale.mat',
            7: 'data/SVHN/svhn_grey_exposure.mat',
            8: 'data/test/train'
        }
        # SELECT FILE:
        self.data_file = self.data_files[1]

        self.ops = array('i', [0, 1, 2, 3, 4, 5, 6, 7])  # Ops [0:+, 1:-, 2:*, 3:/, 4:sin, 5:e, 6:ln, 7:conditional]
        self.pop_size = 160
        self.generations = 10
        self.graph_step = 500
        self.graph_save_step = self.graph_step
        self.json_save_step = 10000
        self.prog_length = 35
        self.num_saved_vals = 50

        self.bid_gp = 1
        self.bid_diff = 0.0001  # Currently in const
        self.check_bid_diff = 1
        self.tangled_graphs = 1
        self.standardize_method = None
        self.selection = const.Selection.BREEDER_MODEL
        self.alpha = 1
        self.use_subset = 1
        self.subset_size = 400
        self.use_validation = 0
        self.validation_size = 1000
        self.train_fitness = fit.fitness_sharing
        # self.train_fitness = fit.avg_detect_rate
        self.test_fitness = fit.avg_detect_rate
        self.subset_sampling = dutil.even_data_subset

        self.grid_sections = 4
        self.test_size = 0.2
        self.breeder_gap = 0.2
        self.var_op_probs = [0.5, 0.5]  # For single-program GP

        self.prob_modify = 0.2
        self.prob_atomic_change = .5
        self.modify_probs = {'action_change': 0.25,
                             'modify': .7,
                             'grid_change': .05}
        self.point_fitness = 1
        self.host_size = self.pop_size
        self.point_gap = 0.1
        self.host_gap = 0.5
        self.prob_removal = 0.7
        self.prob_add = 0.7

        self.limit_atomic_actions = 0
        self.atomic_per_host = 2

        self.prob_mutate = 1
        self.prob_swap = 1
        self.max_teamsize = 10
        self.min_teamsize = 2
        self.start_teamsize = 2
        self.max_start_teamsize = 5

        # Will be calculated/set once data loads
        self.data_shape = None
        self.curr_generation = 0  # Will update when loading other results

        # Components to graph
        self.file_prefix = None
        self.to_graph = {
            'top_trainfit_in_trainset': 0,  # Top training fitness value in training set
            'train_means': 1,  # Mean of training fitness values in training set
            'test_means': 1,  # Mean of testing fitness values in training set
            'top_train_prog_on_test': 1,  # Testing fitness on test set of top training fitness prog in training set
            'top_test_fit_on_train': 1,  # Testing fitness on train set of top training fitness prog in training set
            'percentages': 1,
            'cumulative_detect_rate': 1,
            'top_team_size': 1,
            'incorr_vals': 1,
        }
        self.save_together = 1

        # Image data dimensions
        self.image_data = {
            self.data_files[4]: (28, 28),
            self.data_files[5]: (28, 28),
            self.data_files[6]: (32, 32),
            self.data_files[7]: (32, 32)
        }

        self.adjust_settings()

    # Verify/adjust settings if combinations are not possible
    def adjust_settings(self):
        if not self.use_subset and self.point_fitness != 0:
            self.point_fitness = 0
            print('Setting point fitness to 0')
        if self.point_fitness:
            self.use_subset = 1
        if self.graph_step > self.generations:
            self.graph_step = self.generations
            print('Setting graph step to {}'.format(self.generations))
        if not self.bid_gp:
            self.to_graph['top_team_size'] = 0
            self.tangled_graphs = 0
            print('Setting tangled_graphs to 0')
        if self.pop_size < self.max_teamsize:
            self.max_teamsize = self.pop_size
            print('Setting max teamsize to {}'.format(self.max_teamsize))
        if self.min_teamsize > self.start_teamsize:
            self.start_teamsize = self.min_teamsize
            print('Setting min teamsize to {}'.format(self.min_teamsize))
        if self.tangled_graphs and self.bid_gp == 0:
            self.bid_gp = 1
            print('Setting bid_gp to 1')
        if self.bid_gp:
            assert self.max_teamsize >= self.min_teamsize

        if self.data_file in self.image_data.keys():
            self.data_shape = self.image_data[self.data_file]
        else:
            self.grid_sections = 0
            print('Setting grid_sections to 0')

        if self.grid_sections == 0 and self.modify_probs['grid_change'] != 0:
            amt = self.modify_probs['grid_change'] / 2
            self.modify_probs['action_change'] += amt
            self.modify_probs['modify'] += amt
            self.modify_probs['grid_change'] = 0
            print('Changed modify probs: \n{}'.format(self.modify_probs))

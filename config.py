import const, data_utils as dutil, fitness as fit
from array import array


class Config:
    def __init__(self):
        self.ops = array('i', [0, 1, 2, 3, 4, 5, 6, 7])  # Ops [0:+, 1:-, 2:*, 3:/, 4:sin, 5:e, 6:ln, 7:conditional]
        self.pop_size = 50
        self.generations = 100
        self.graph_step = 200
        self.graph_save_step = 200
        self.json_save_step = None
        self.prog_length = 24
        self.num_saved_vals = 50
        self.data_files = ['data/iris.data', 'data/tic-tac-toe.data', 'data/ann-train.data', 'data/shuttle.trn',
                           'data/MNIST', 'data/gisette_train.data']
        self.data_file = self.data_files[3]
        self.standardize_method = const.StandardizeMethod.MEAN_VARIANCE
        self.selection = const.Selection.BREEDER_MODEL
        self.breeder_gap = 0.2
        self.alpha = 1
        self.use_subset = 1
        self.subset_size = 200
        self.use_validation = 1
        self.validation_size = self.subset_size
        # self.use_generator = 0
        self.test_size = 0.2
        # self.train_fitness = fit.fitness_sharing
        self.train_fitness = fit.avg_detect_rate
        self.test_fitness = fit.avg_detect_rate
        self.subset_sampling = dutil.even_data_subset

        self.var_op_probs = [0.5, 0.5]
        self.action_change_probs = [0.5, 0.5]
        self.bid_gp = 1
        self.point_fitness = 1
        self.host_size = self.pop_size
        self.point_gap = 0.2
        self.host_gap = 0.5
        self.prob_removal = 0.7
        self.prob_add = 0.7
        self.prob_modify = 0.2
        self.prob_change_action = 0.1
        self.prob_mutate = 1
        self.prob_swap = 1
        self.max_teamsize = 5
        self.min_teamsize = 2
        self.start_teamsize = 2
        self.tangled_graphs = 1

        # Will be calculated/set once data loads
        self.num_ipregs = None
        self.output_dims = None
        self.max_vals = []

        # Components to graph
        self.file_prefix = None
        self.to_graph = {
            'top_trainfit_in_trainset': 0,  # Top training fitness value in training set
            'train_means': None,  # Mean of training fitness values in training set
            'test_means': 1,  # Mean of testing fitness values in training set
            'top_train_prog_on_test': 1,  # Testing fitness on test set of top training fitness prog in training set
            'top_test_fit_on_train': 1,  # Testing fitness on train set of top training fitness prog in training set
            'percentages': 1,
            'cumulative_detect_rate': 1,
            'top_team_size': 1,
        }
        self.to_graph['train_means'] = self.to_graph['top_trainfit_in_trainset']

        self.adjust_settings()

    # Verify/adjust settings if combinations are not possible
    def adjust_settings(self):
        if not self.use_subset:
            self.point_fitness = 0
        if self.graph_step > self.generations:
            self.graph_step = self.generations
        if not self.bid_gp:
            self.to_graph['top_team_size'] = 0
            self.tangled_graphs = 0
        if self.pop_size < self.max_teamsize:
            self.max_teamsize = self.pop_size
        if self.min_teamsize > self.start_teamsize:
            self.start_teamsize = self.min_teamsize
        if self.tangled_graphs:
            self.bid_gp = 1

        if self.bid_gp:
            assert self.max_teamsize >= self.min_teamsize

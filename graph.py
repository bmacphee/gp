import matplotlib, os, time, pdb
import const, fitness as fit, numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


def make_graphs(graph_iter, env, data, stats, pop, hosts, hosts_i):
    graph_params = stats.get_graph_params(env.to_graph)
    graph_fitness(graph_iter, data, env, *graph_params)
    if env.to_graph['percentages']:
        graph_percs(graph_iter, stats.percentages, env)
    if env.to_graph['cumulative_detect_rate']:
        cumulative = fit.cumulative_detect_rate(data, pop, hosts, stats.trainset_with_testfit, hosts_i=hosts_i)
        graph_cumulative(env, cumulative)
    if env.to_graph['top_team_size']:
        graph_teamsize(graph_iter, env, stats.num_progs_per_top_host)
    plt.close('all')


def graph_cumulative(env, cumulative):
    fig = plt.figure(figsize=(13, 13), dpi=80)
    ax = fig.add_subplot(111)
    num = len(cumulative)
    ax.plot(range(1, num + 1), cumulative, linewidth=2)
    plt.title('Cumulative Class-Wise Detection Rate (Training data: {})'.format(env.data_file))
    ax.set_xlim(1, num)
    ax.set_ylim(0, 1.02)
    filename = '{}{}'.format(env.file_prefix, const.FILE_NAMES[graph_cumulative.__name__])
    save_figure(filename, fig)


def graph_teamsize(last_x, env, team_sizes):
    gens = [i * env.graph_step for i in range(last_x)]
    fig = plt.figure(figsize=(13, 13), dpi=80)
    ax = fig.add_subplot(111)
    ax.plot(gens, team_sizes, label='# progs', linewidth=2)
    plt.title(
        '# Programs in Top Host\nMax Team Size: {}\nProgram Length: {}'.format(env.max_teamsize, env.prog_length))
    if gens[-1] != 0:
        ax.set_xlim(xmax=gens[-1])
    ax.set_xlim(0)
    ax.set_ylim(0, env.max_teamsize)
    filename = '{}{}'.format(env.file_prefix, const.FILE_NAMES[graph_teamsize.__name__])
    save_figure(filename, fig)


def graph_fitness(last_x, data, env, top_train_fit_on_train=None, train_means=None, test_means=None,
                  top_train_prog_on_test=None, top_test_fit_on_train=None):
    gens = [i * env.graph_step for i in range(last_x)]
    fig = plt.figure(figsize=(13, 13), dpi=80)
    ax = fig.add_subplot(111)
    valid_train = 'Validation' if env.use_validation else 'Train'
    labels = [
        'Max Train Fitness in Train Set', 'Mean Train Fitness in Train Set', 'Mean Test Fitness in Train Set',
        'Best {} Prog on Test Set (using test_fit)'.format(valid_train),
        'Best Train Prog on {} Set (using test_fit)'.format(valid_train)
    ]

    if top_train_fit_on_train:
        ax.plot(gens, top_train_fit_on_train, label=labels[0], linewidth=2)
    if train_means:
        ax.plot(gens, train_means, label=labels[1], linewidth=2)
    if test_means:
        ax.plot(gens, test_means, label=labels[2], linewidth=2)
    if top_test_fit_on_train:
        ax.plot(gens, top_test_fit_on_train, label=labels[4], linewidth=2)
    if top_train_prog_on_test:
        ax.plot(gens, top_train_prog_on_test, label=labels[3], linewidth=2)

    subset_str = ', Subset Size: {}'.format(data.act_subset_size) if env.use_subset else ''
    valid_str = ''
    if env.use_validation:
        valid_str = ', Validation Size: {}'.format(data.act_valid_size)
    op_str = ', '.join([const.OPS[x] for x in env.ops])
    plt.title(
        'Data: {}\nSelection: {}, Bid GP: {}, Point Fitness: {}, Graphs: {}\nPop Size: {}, Generations: {}, Step Size: {}{}{}\n'
        'Training Fitness: {}, Test Fitness: {}\nOps: [{}], Alpha: {}'.format(env.data_file, env.selection.value,
                                                                              env.bid_gp, env.point_fitness,
                                                                              env.tangled_graphs, env.pop_size,
                                                                              env.generations, env.graph_step,
                                                                              subset_str, valid_str,
                                                                              env.train_fitness.__name__,
                                                                              env.test_fitness.__name__, op_str,
                                                                              env.alpha))
    if gens[-1] != 0:
        ax.set_xlim(xmax=gens[-1])
    ax.set_xlim(0)
    ax.set_ylim(0, 1.02)
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    filename = '{}{}'.format(env.file_prefix, const.FILE_NAMES[graph_fitness.__name__])
    save_figure(filename, fig)


def graph_percs(last_x, percentages, env):
    generations = [i * env.graph_step for i in range(last_x)]
    fig = plt.figure(figsize=(13, 13), dpi=80)
    ax = fig.add_subplot(111)
    labels = sorted([perc for perc in percentages])
    for l in labels:
        ax.plot(generations, percentages[l], label=l, linewidth=2)
    plt.title('% Classes Correct (Training data: {})'.format(env.data_file))
    if generations[-1] != 0:
        ax.set_xlim(xmax=generations[-1])
    ax.set_xlim(0)
    ax.set_ylim(0, 1.02)
    plt.legend(bbox_to_anchor=(1.1, 1), fontsize=8)
    filename = '{}{}'.format(env.file_prefix, const.FILE_NAMES[graph_percs.__name__])
    save_figure(filename, fig)


def save_figure(filename, fig):
    ax = fig.axes[0]
    plt.grid(which='both', axis='both')

    if filename.find(const.FILE_NAMES['graph_cumulative']) != -1:
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    if filename.find(const.FILE_NAMES['graph_teamsize']) != -1:
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_label_position('right')
    else:
        ax.yaxis.set_major_locator(MultipleLocator(.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.tick_params(which='both', width=1)

    date = time.strftime("%d_%m_%Y")
    filepath = os.path.join(const.IMAGE_DIR, date, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.set_size_inches(22, 11)
    fig.savefig(filepath, dpi=100)

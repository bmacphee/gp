import matplotlib, os, time, numpy as np, pdb
import const, fitness as fit, utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


def make_graphs(env, data, stats, system):
    pop, hosts, hosts_i = system.pop, system.hosts, system.root_hosts
    graph_iter = len(stats.testset_with_testfit)
    graph_params = stats.get_graph_params(env.to_graph)
    num_plots = len([key for key in env.to_graph.keys() if env.to_graph[key]])
    # Graphs that are combined into 1 figure (to calculate number of figures needed)
    combined = sum([env.to_graph['top_trainfit_in_trainset'], env.to_graph['train_means'],
                    env.to_graph['top_train_prog_on_test'], env.to_graph['top_test_fit_on_train'],
                    env.to_graph['test_means']])
    num_plots -= (combined - 1)

    if env.save_together:
        # Save all graphs in 1 figure
        cols = 2 if num_plots != 1 else 1
        rows = int(num_plots / cols) + (num_plots % cols)
        fig, axes = plt.subplots(rows, cols)
        for a in axes.flatten():
            a.grid(which='both')
        axes = axes.flatten()
    else:
        # Graphs will be saved individually
        fig = plt.figure()
        axes = [None] * num_plots
        plt.grid()

    plot_count = 0
    graph_fitness(graph_iter, data, env, *graph_params, ax=axes[plot_count])
    plot_count += 1

    if env.to_graph['cumulative_detect_rate']:
        cumulative = fit.cumulative_detect_rate(data, pop, hosts, stats.trainset_with_testfit, hosts_i=hosts_i)
        graph_cumulative(env, cumulative, ax=axes[plot_count], save=not env.save_together)
        plot_count += 1
    if env.to_graph['percentages']:
        graph_percs(graph_iter, stats.percentages, env, ax=axes[plot_count], save=not env.save_together)
        plot_count += 1
    if env.to_graph['incorr_vals']:
        graph_incorr_vals(stats.last_y_pred, data.y_test, data.classes, env, ax=axes[plot_count],
                          save=not env.save_together)
        plot_count += 1
    if env.to_graph['top_team_size']:
        graph_teamsize(graph_iter, env, stats, ax=axes[plot_count], save=not env.save_together)
        plot_count += 1
    if env.save_together:
        plt.subplots_adjust(hspace=0.2, wspace=0.1)
        save_figure('all', env.file_prefix, fig)
    plt.close('all')


def graph_cumulative(env, cumulative, ax=None, save=0):
    num = len(cumulative)
    ax = plot(range(1, num + 1), cumulative, None, ax)
    ax.set_title('Cumulative Class-Wise Detection Rate (Training data: {})'.format(env.data_file))
    ax.tick_params(which='both', width=1, labeltop=False, labelright=True)
    ax.set_xlim(1, num)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    if save:
        save_figure(const.FILE_NAMES[graph_cumulative.__name__], env.file_prefix, plt.gcf())


def graph_teamsize(last_x, env, stats, ax=None, save=0):
    team_sizes = stats.num_progs_per_top_host
    gens = [i * env.graph_step for i in range(last_x)]
    if not env.tangled_graphs:
        av_str = ''
    else:
        av_str = 'aver.'

    ax = plot(gens, team_sizes, '{}# progs'.format(av_str), ax)
    ax.set_ylim(0, env.max_teamsize)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.legend(fontsize=8)

    if not env.tangled_graphs:
        # Graph just program size in top host if tangled_graphs not used
        active_progs = stats.active_progs_per_top_host
        ax = plot(gens, active_progs, '# active progs', ax)
    else:
        # If using tangled_graphs, then graph hosts per top solution and average number of programs per host
        num_hosts_in_graph = stats.num_hosts_in_graph
        ax2 = ax.twinx()
        ax2 = plot(gens, num_hosts_in_graph, '# hosts in graph', ax=ax2, color='red')
        ax2.set_ylim(0)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines+lines2, labels+labels2, fontsize=8)

    ax.set_title(
        '# Programs in Top Host (Max Team Size: {}, Program Length: {})'.format(env.max_teamsize, env.prog_length))
    if gens[-1] != 0:
        ax.set_xlim(xmax=gens[-1])
    ax.tick_params(which='both', width=1)
    ax.set_xlim(0)

    if save:
        save_figure(const.FILE_NAMES[graph_teamsize.__name__], env.file_prefix, plt.gcf())


def graph_fitness(last_x, data, env, top_train_fit_on_train=None, train_means=None, test_means=None,
                  top_train_prog_on_test=None, top_test_fit_on_train=None, ax=None, save=0):
    gens = [i * env.graph_step for i in range(last_x)]
    valid_train = 'Validation' if env.use_validation else 'Train'
    prog_host = 'Prog' if not env.bid_gp else 'Host'
    labels = [
        'Max Train Fitness in Train Set', 'Mean Train Fitness in Train Set', 'Mean Test Fitness in Train Set',
        'Best {} {} on Test Set (with test_fit)'.format(valid_train, prog_host),
        'Best Train {} on {} Set (with test_fit)'.format(prog_host, valid_train)
    ]

    if top_train_fit_on_train:
        ax = plot(gens, top_train_fit_on_train, labels[0], ax)
    if train_means:
        ax = plot(gens, train_means, labels[1], ax)
    if test_means:
        ax = plot(gens, test_means, labels[2], ax)
    if top_test_fit_on_train:
        ax = plot(gens, top_test_fit_on_train, labels[4], ax)
    if top_train_prog_on_test:
        ax = plot(gens, top_train_prog_on_test, labels[3], ax)

    subset_str = ', Subset Size: {}'.format(data.act_subset_size) if env.use_subset else ''
    valid_str, grid_str = '', ''
    if env.use_validation:
        valid_str = ', Validation Size: {}'.format(data.act_valid_size)
    if env.grid_sections:
        grid_str = ', Grid Sections: {}'.format(env.grid_sections)
    op_str = ', '.join([const.OPS[x] for x in env.ops])
    ax.set_title(
        'Data: {}\nSelection: {}, Bid GP: {}, Point Fitness: {}, Graphs: {}\nPop Size: {}, Generations: {}, Step Size: '
        '{}{}{}{}\nTraining Fitness: {}, Test Fitness: {}\nOps: [{}]'.format(env.data_file, env.selection.value,
                                                                             env.bid_gp, env.point_fitness,
                                                                             env.tangled_graphs, env.pop_size,
                                                                             env.generations, env.graph_step,
                                                                             subset_str,
                                                                             valid_str, grid_str,
                                                                             env.train_fitness.__name__,
                                                                             env.test_fitness.__name__, op_str)
    )
    if gens[-1] != 0:
        ax.set_xlim(xmax=gens[-1])
    ax.tick_params(which='both', width=1, labeltop=False, labelright=True)
    ax.set_xlim(0)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc=0, fontsize=8)

    if save:
        save_figure(const.FILE_NAMES[graph_fitness.__name__], env.file_prefix, plt.gcf())


def graph_percs(last_x, percentages, env, ax=None, save=0):
    generations = [i * env.graph_step for i in range(last_x)]
    labels = sorted([perc for perc in percentages])
    for l in labels:
        ax = plot(generations, percentages[l], l, ax)
    ax.set_title('% Classes Correct (Training data: {})'.format(env.data_file))
    if generations[-1] != 0:
        ax.set_xlim(xmax=generations[-1])
    ax.tick_params(which='both', width=1, labeltop=False, labelright=True)
    ax.set_xlim(0)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc=0, fontsize=8)
    if save:
        save_figure(const.FILE_NAMES[graph_percs.__name__], env.file_prefix, plt.gcf())


def graph_incorr_vals(y_pred, y_act, classes, env, ax=None, save=0):
    if ax is None:
        ax = plt.gca()

    sorted_classes = sorted(classes)
    results = {cl: [] for cl in sorted_classes}
    for cl in sorted_classes:
        cl_results = y_pred[[i for i in range(len(y_act)) if y_act[i] == classes[cl] and y_pred[i] != y_act[i]]]
        for c in sorted_classes:
            results[cl].append(len(np.where(cl_results == classes[c])[0]))

    ind = np.arange(len(classes))
    res = np.array([results[i] for i in sorted_classes])
    ax.bar(ind, res[:, 0], label=sorted_classes[0])
    sum = res[:, 0]
    for i in range(1, len(classes)):
        ax.bar(ind, res[:, i], bottom=sum, label=sorted_classes[i])
        sum += res[:, i]
    ax.legend()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    sorted_classes.insert(0, '')
    ax.set_xticklabels(sorted_classes)

    if save:
        save_figure('incorr_vals', env.file_prefix, plt.gcf())


def plot(x, y, label, ax=None, color=None):
    if ax is None:
        ax = plt.gca()
    if color:
        ax.plot(x, y, label=label, linewidth=2, color=color)
    else:
        ax.plot(x, y, label=label, linewidth=2)
    return ax


def save_figure(filename, file_prefix, fig):
    filepath = utils.make_filename(const.IMAGE_DIR, file_prefix, filename)
    fig.set_size_inches(22, 11)
    fig.savefig(filepath, dpi=100)
    fig.clear()

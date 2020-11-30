import os
import math
import json
import time

from os import path

import numpy as np



def get_results(exp):
    report_path = path.join(exp, "results")
    results = []
    for file in os.listdir(report_path):
        if path.isfile(path.join(report_path, file)):
            with open(path.join(report_path, file), 'r') as f:
                results.append(json.load(f))
    return results


def outliers_idx(values, max_deviation):
    mean = values.mean()

    dist = abs(values - mean)
    corrected = dist < mean + max_deviation
    return corrected


def logscale_plot(ax, xrange, yrange, base=10):

    from matplotlib import ticker

    if xrange is not None:
        ax.xaxis.set_minor_formatter(ticker.LogFormatter())
        ax.xaxis.set_major_formatter(ticker.LogFormatter())
        ax.set_xscale("log", base=base)
        ax.set_xlim([np.min(xrange), np.max(xrange)])
    if yrange is not None:
        ax.yaxis.set_minor_formatter(ticker.LogFormatter())
        ax.yaxis.set_major_formatter(ticker.LogFormatter())
        ax.set_yscale("log", base=base)
        ax.set_ylim([yrange.min() - 0.1*yrange.min(), yrange.max() + 0.1*yrange.min()])


def scale(x):
    if len(x) == 0:
        print("WARNING : The array of scores is empty")
        return x

    range_of_x_values = x.ptp()
    if (range_of_x_values != 0):
        return (x - x.min()) / (x.ptp())
    else:
        return x/x.min() #an array of 1 if all the values are identical



def cross_parameter_plot(ax, values, scores, loss, smaxs, cmaxs, lmaxs, p1, p2, log1, log2, cat1, cat2, log_loss_color_scale):

    X = values[p2].copy()
    Y = values[p1].copy()

    to_log = []
    if log1 and not cat2:
        to_log.append(X)
    else:
        to_log.append(None)
        if cat2:
            ax.margins(x=0.05)
    if log1 and not cat1:
        to_log.append(Y)
    else:
        to_log.append(None)

    logscale_plot(ax, *to_log)
    # if log2 and not cat2:
    #     logscale_plot(ax, None, Y)

    ax.tick_params(axis='both', which='both')
    ax.tick_params(axis='both', labelsize="xx-small")
    ax.grid(True, which="both", ls="-", alpha=0.75)

    ax.set_xlabel(p2)
    ax.set_ylabel(p1)

    if log_loss_color_scale:
        sc_l = ax.scatter(X[lmaxs], Y[lmaxs], scores[lmaxs]*100, c=np.log(loss[lmaxs]), cmap="inferno") #the color scale of the loss is logarithmic
    else:
        sc_l = ax.scatter(X[lmaxs], Y[lmaxs], scores[lmaxs]*100, c=loss[lmaxs], cmap="inferno")

    sc_s = ax.scatter(X[smaxs], Y[smaxs], scores[smaxs]*100, c=cmaxs, cmap="YlGn")
    sc_m = ax.scatter(X[~(lmaxs)], Y[~(lmaxs)], scores[~(lmaxs)]*100, color="red")

    return sc_l, sc_s, sc_m


def loss_plot(ax, values, scores, loss, smaxs, cmaxs, lmaxs, p, log, categorical, legend, loss_behaviour):

    X = values[p].copy()

    if log and not(categorical):
        logscale_plot(ax, X, loss)
    else:
        logscale_plot(ax, None, loss)
        if categorical:
            ax.margins(x=0.05)

    ax.set_xlabel(p)
    ax.set_ylabel("loss")

    ax.tick_params(axis='both', which='both')
    ax.tick_params(axis='both', labelsize="xx-small")
    ax.grid(True, which="both", ls="-", alpha=0.75)

    sc_l = ax.scatter(X[lmaxs], loss[lmaxs], scores[lmaxs]*100, color="orange")
    sc_s = ax.scatter(X[smaxs], loss[smaxs], scores[smaxs]*100, c=cmaxs, cmap="YlGn")
    if loss_behaviour == "min":
        sc_m = ax.scatter(X[~(lmaxs)], [loss.min()]*np.sum(~lmaxs), scores[~(lmaxs)]*100, color="red", label="Loss min.")
    else:
        sc_m = ax.scatter(X[~(lmaxs)], [loss.max()]*np.sum(~lmaxs), scores[~(lmaxs)]*100, color="red", label="Score max.")

    if legend:
        ax.legend()

    return sc_l, sc_s, sc_m


def parameter_violin(ax, values, scores, loss, smaxs, cmaxs, p, log, legend):

    import matplotlib.pyplot as plt

    y = values[p].copy()[smaxs]
    all_y = values[p].copy()

    if log:
        y = np.log10(y)
        all_y = np.log10(all_y)

    ax.get_yaxis().set_ticks([])
    ax.tick_params(axis='x', which='both')

    ax.xaxis.set_major_locator(plt.MultipleLocator(1))

    def format_func(value, tick_number):
        return "$10^{"+str(int(np.floor(value)))+"}$"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    ax.set_xlabel(p)
    ax.grid(True, which="both", ls="-", alpha=0.75)
    ax.set_xlim([np.min(all_y), np.max(all_y)])

    violin = ax.violinplot(y, vert=False, showmeans=False, showextrema=False)

    for pc in violin['bodies']:
        pc.set_facecolor('forestgreen')
        pc.set_edgecolor('white')

    quartile1, medians, quartile3 = np.percentile(y, [25, 50, 75])

    ax.scatter(medians, 1, marker='o', color='orange', s=30, zorder=4, label="Median")
    ax.hlines(1, quartile1, quartile3, color='gray', linestyle='-', lw=4, label="Q25/Q75")
    ax.vlines(y.mean(), 0.5, 1.5, color='blue', label="Mean")

    ax.scatter(np.log10(values[p][scores.argmax()]), 1, color="red", zorder=5, label="Best score")
    ax.scatter(y, np.ones_like(y), c=cmaxs, cmap="YlGn", alpha=0.5, zorder=3)

    if legend:
        ax.legend(loc=2)


def pretty_value(x):
    """If x is an integer float, it return the int x,
    else return x with maximum 2 decimals. To get pretty
    number value for the legends of the plots"""
    if (x-int(x) == 0):
        return int(x)
    else:
        return round(x,2)


def parameter_bar(ax, values, scores, loss, smaxs, cmaxs, p, categories):

    import matplotlib.pyplot as plt

    y = values[p].copy()[smaxs]

    ax.set_xlabel(p)
    ax.grid(True, which="both", ls="-", alpha=0.75)

    heights = []
    for p in categories:
        heights.append(y.tolist().count(p))

    bar = ax.bar(x=categories, height=heights, color="forestgreen", alpha=0.3)


def plot_hyperopt_report(exp, params, metric='loss', loss_metric="loss",
                         loss_behaviour="min", not_log=None, categorical=None,
                         max_deviation=None, title=None, use_log_score=False,
                         show_minimal_scores=True, best_proportion_to_show=0.05,
                         log_loss_color_scale=False):
    """Cross paramater scatter plot of hyperopt trials.

    Installation of Matplotlib and Seaborn packages is required to use this tool.

    Arguments:
        exp {str or Path} -- Report directory storing hyperopt trials results.
        params {list} -- Parameters to plot.

    Keyword Arguments:
        metric {str} -- Metric to use as performance measure, stored in the hyperopt trials results dictionnaries.
                        May be different from loss metric. By default, loss is used as performance metric. (default: 'loss')
        not_log {list} -- Parameters to plot with a linear scale. By default, all scales are logarithmic.
        categorical {list} -- List of categorical parameters. (default: None)
        max_deviation {float} -- Maximum standard deviation expected from the loss mean. Useful to remove outliers.
                                 (default: {None})
        use_log_score {bool} -- If true, the logarithm of the score will be use for the plots (default : {False})
        log_loss_color_scale {bool} -- If true, the color scale showing the loss will be logarithmic (default : {False})
        show_minimal_scores {bool} -- Whether to put forward minimal values of the score or maximal values (default : {True})
        best_proportion_to_show {float} -- The proportion of the scores that should be put forward, must be in the range [0,1]. (default : {0.05})

        title {str} -- Optional title for the figure. (default: {None})

    Returns:
        [matplotlib.pyplot.figure] -- Matplotlib figure object.

    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(context="paper", style="darkgrid", font_scale=1.5)
    N = len(params)
    not_log = not_log or []

    results = get_results(exp)

    loss = np.array([r['returned_dict'][loss_metric] for r in results])
    scores = np.array([r['returned_dict'][metric] for r in results])

    if use_log_score:
        scores = np.log(scores)

    if max_deviation is not None:
        not_outliers = outliers_idx(loss, max_deviation)
        loss = loss[not_outliers]
        scores = scores[not_outliers]
        values = {p: np.array([r['current_params'][p] for r in results])[not_outliers] for p in params}
    else:
        values = {p: np.array([r['current_params'][p] for r in results]) for p in params}

    scores = scale(scores) # Rescale the scores linearly to a range of [0,1]
    if show_minimal_scores:
        # We invert the order of the score value so that
        # the bigger values correspond to the minimal orginal
        # scores. They are the new maximal ones that will be put forward.
        scores = 1 - scores

    categorical = categorical or []

    for p in categorical:
        values[p] = values[p].astype(str)

    # Sorting for categorical plotting
    all_categorical = [val for p, val in values.items() if p in categorical]
    all_numerical = [val for p, val in values.items() if p not in categorical]

    sorted_idx = np.lexsort((loss, scores, *all_numerical, *all_categorical))

    loss = np.array([loss[i] for i in sorted_idx])
    scores = np.array([scores[i] for i in sorted_idx])

    for p, val in values.items():
        values[p] = np.array([val[i] for i in sorted_idx])

    ## loss and f1 values

    if loss_behaviour == "min":
        lmaxs = loss > loss.min()
    else:
        lmaxs = loss < loss.max()

    c_l = np.log10(loss[lmaxs])

    percent = math.ceil(len(scores) * best_proportion_to_show)
    smaxs = scores.argsort()[-percent:][::-1]
    cmaxs = scale(scores[smaxs])

    ## gridspecs

    fig = plt.figure(figsize=(15, 19), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2/30, 28/30])
    fig.suptitle(f"Hyperopt trials summary - {title}", size=15)

    gs0 = gs[0].subgridspec(1, 3)
    gs1 = gs[1].subgridspec(N + 1, N)


    lbar_ax = fig.add_subplot(gs0[0, 0])
    fbar_ax = fig.add_subplot(gs0[0, 1])
    rad_ax = fig.add_subplot(gs0[0, 2])
    rad_ax.axis('off')

    # plot
    axes = []
    for i, p1 in enumerate(params):
        for j, p2 in enumerate(params):
            ax = fig.add_subplot(gs1[i, j])
            axes.append(ax)
            if p1 == p2:
                sc_l, sc_s, sc_m = loss_plot(ax, values, scores, loss, smaxs,
                                             cmaxs, lmaxs, p2, not(p2 in not_log),
                                             p2 in categorical, (i==0 and j==0),
                                             loss_behaviour)
            else:
                sc_l, sc_s, sc_m = cross_parameter_plot(ax, values, scores, loss,
                                     smaxs, cmaxs, lmaxs, p1, p2,
                                     not(p1 in not_log), not(p2 in not_log),
                                     p1 in categorical, p2 in categorical,
                                     log_loss_color_scale)

    #legends

    handles, labels = sc_l.legend_elements(prop="sizes")

    if use_log_score:
        legend = rad_ax.legend(handles, labels, loc="center left",
                            title=f"Normalized log of {metric} (%)", mode='expand',
                            ncol=len(labels) // 2 + 1)
    else:
        legend = rad_ax.legend(handles, labels, loc="center left",
                            title=f"Normalized {metric} (%)", mode='expand',
                            ncol=len(labels) // 2 + 1)


    l_cbar = fig.colorbar(sc_l, cax=lbar_ax, ax=axes, orientation="horizontal")

    if log_loss_color_scale:
        _ = l_cbar.ax.set_title("Loss value (logarithmic color scale)")
    else:
        _ = l_cbar.ax.set_title("Loss value")

    f_cbar = fig.colorbar(sc_s, cax=fbar_ax, ax=axes,
                          orientation="horizontal", ticks=[0, 0.5, 1])
    _ = f_cbar.ax.set_title(f"{metric} best population")
    _ = f_cbar.ax.set_xticklabels([f"{pretty_value(best_proportion_to_show*100)}% best",
                                   f"{pretty_value(best_proportion_to_show*50)}% best", "Best"])

    #violinplots

    legend = True
    for i, p in enumerate(params):
        ax = fig.add_subplot(gs1[-1, i])
        if p in categorical:
            parameter_bar(ax, values, scores,
                          loss, smaxs, cmaxs, p, sorted(list(set(values[p]))))
        else:
            parameter_violin(ax, values, scores,
                             loss, smaxs, cmaxs, p, not(p in not_log), legend)
            legend = False
        if legend:
            ax.set_ylabel(f"{pretty_value(best_proportion_to_show*100)}% best {metric}\nparameter distribution")
    return fig
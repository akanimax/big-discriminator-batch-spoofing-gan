""" Script for generating the plot of the INCEPTION SCORE values (with max annotated) """

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def annot_max(x, y, ax=None, y_offset=0.96):
    """
    highlight the max of a plot automatically:
    :param x: x values
    :param y: y values
    :param ax: matplotlib axis object
    :param y_offset: the y offset of the place where maxs are annotated
    :return: None
    """
    xmax = x[np.argmax(y)]
    ymax = max(y)
    text = "epoch=%d, IS=%.3f" % (xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, y_offset), **kw)

    # make the font size of the annotation smaller
    for child in ax.get_children():
        if isinstance(child, matplotlib.text.Annotation):
            child.set_fontsize(10)


def read_log_file(file_path):
    """
    reads the values from a log_file
    :param file_path: path to the log file
    :return: indices, values => indices and values for them
    """
    indices, values = [], []  # initialize these to empty lists
    with open(file_path, "r") as filer:
        for line in filer:
            ind, val = line.strip().split("\t")
            ind, val = int(ind), float(val)
            indices.append(ind)
            values.append(val)
    return indices, values


def generate_plot(xs, ys, titles, save_path):
    """
    generates the plot given the indices and is values
    :param xs: the indices (epochs)
    :param ys: IS values
    :param titles: title of the generated plot
    :param save_path: path to save the file
    :return: None (saves file)
    """
    font = {'family': 'normal', 'size': 20}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(10, 6))

    plt.xlabel("Epochs")
    plt.ylabel("Inception scores")

    # set the y limit to 4 + max of everything
    plt.ylim(0, max(map(max, ys)) + 5)

    for cnt, x, y, title in zip(range(len(xs)), xs, ys, titles):
        annot_max(x, y, y_offset=0.96 - (0.07 * cnt))
        plt.margins(.05, .05)
        plt.plot(x, y, linewidth=4, label=title)

    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')


def parse_arguments():
    """
    default command line parser
    :return: args => parsed commandline arguments
    """

    parser = argparse.ArgumentParser("Inception Score plot generator")

    parser.add_argument("--log_files", action="store", type=str,
                        default=None,
                        help="paths to the inception score log files",
                        nargs="+")

    parser.add_argument("--plot_titles", action="store", type=str,
                        default=None,
                        help="titles of the plots for the logs",
                        nargs="+")

    parser.add_argument("--save_path", action="store", type=str,
                        default=None,
                        help="path to save the generated_plots")

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function of the script
    :param args: parsed command line arguments
    :return: None
    """
    assert len(args.log_files) == len(args.plot_titles), \
        "Incompatible number of log files and their titles"

    xs, ys = [], []  # initialize to empty lists

    for log_file in args.log_files:
        x, y = read_log_file(log_file)
        xs.append(x)
        ys.append(y)

    generate_plot(xs, ys, args.plot_titles, args.save_path)


if __name__ == '__main__':
    main(parse_arguments())

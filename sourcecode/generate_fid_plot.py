""" Script for generating the plot of the FID values (with min annotated) """

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def annot_min(x, y, ax=None):
    """
    highlight the min of a plot automatically:
    :param x: x values
    :param y: y values
    :param ax: matplotlib axis object
    :return: None
    """
    xmin = x[np.argmin(y)]
    ymin = min(y)
    text = "epoch=%d, FID=%.3f" % (xmin, ymin)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.94, 0.96), **kw)


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


def generate_plot(x, y, title, save_path):
    """
    generates the plot given the indices and fid values
    :param x: the indices (epochs)
    :param y: fid values
    :param title: title of the generated plot
    :param save_path: path to save the file
    :return: None (saves file)
    """
    font = {'family': 'normal', 'size': 20}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(10, 6))
    annot_min(x, y)
    plt.margins(.05, .05)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("FID scores")
    plt.plot(x, y, linewidth=4)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')


def parse_arguments():
    """
    default command line parser
    :return: args => parsed commandline arguments
    """

    parser = argparse.ArgumentParser("FID plot generator")

    parser.add_argument("--log_file", action="store", type=str,
                        default=None,
                        help="path to the fid log file")

    parser.add_argument("--plot_title", action="store", type=str,
                        default=None,
                        help="title of the plot used")

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
    generate_plot(*read_log_file(args.log_file),
                  args.plot_title, args.save_path)


if __name__ == '__main__':
    main(parse_arguments())

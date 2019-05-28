import matplotlib.pyplot as plt
import csv
import os
import matplotlib
import argparse
import numpy as np
from scripts.plot_configs import Configs
from matplotlib import cm
matplotlib.use('Agg')  # in order to be able to save figure through ssh
# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode


def configure_plot(fontsize, usetex):
    fontsize = fontsize
    matplotlib.rc("text", usetex=usetex)
    matplotlib.rcParams['axes.linewidth'] = 0.1
    matplotlib.rc("font", family="Times New Roman")
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = "Times"
    matplotlib.rcParams["figure.figsize"] = 10, 8
    matplotlib.rc("xtick", labelsize=fontsize)
    matplotlib.rc("ytick", labelsize=fontsize)


def truncate_to_same_len(arrs):
    min_len = np.min([x.size for x in arrs])
    arrs_truncated = [x[:min_len] for x in arrs]
    return arrs_truncated


def read_attr(csv_path, attr):
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter='\t')
        try:
            row = next(reader)
        except Exception:
            return None
        if attr not in row:
            return None
        idx = row.index(attr)  # the column number for this attribute
        vals = []
        for row in reader:
            vals.append(row[idx])
    return np.array(vals, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir_parent', help='The parent dir for experiments', type=str)
    parser.add_argument('--value', help='column name in the log.txt file', type=str)
    parser.add_argument('--output', type=str, default='', help='output file name')
    parser.add_argument('--expert_performance', type=float)

    parser.add_argument('--style', type=str, default='', help='icml_piccolo_final')
    parser.add_argument('--y_higher', nargs='?', type=float)
    parser.add_argument('--y_lower', nargs='?', type=float)
    parser.add_argument('--n_iters', nargs='?', type=int)
    parser.add_argument('--legend_loc', type=int, default=0)
    parser.add_argument('--curve', type=str, default='percentile', help='percentile, std')

    args = parser.parse_args()
    args.attr = args.value
    args.dir = args.logdir_parent
    conf = Configs(args.style)
    subdirs = sorted(os.listdir(args.dir))
    subdirs = [d for d in subdirs if d[0] != '.']  # filter out weird things, e.g. .DS_Store
    subdirs = conf.sort_dirs(subdirs)
    fontsize = 32 if args.style else 12  # for non style plots, exp name can be quite long
    usetex = True if args.style else False
    configure_plot(fontsize=fontsize, usetex=usetex)
    linewidth = 4
    n_curves = 0
    for exp_name in subdirs:
        exp_dir = os.path.join(args.dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        data = []
        for root, _, files in os.walk(exp_dir):
            if 'log.txt' in files:
                d = read_attr(os.path.join(root, 'log.txt'), args.attr)
                if d is not None:
                    data.append(d)
        if not data:
            continue
        n_curves += 1
        data = np.array(truncate_to_same_len(data))
        if args.curve == 'std':
            mean, std = np.mean(data, axis=0), np.std(data, axis=0)
            low, mid, high = mean - std, mean, mean + std
        elif args.curve == 'percentile':
            low, mid, high = np.percentile(data, [25, 50, 75], axis=0)
        if args.n_iters is not None:
            mid, high, low = mid[:args.n_iters], high[:args.n_iters], low[:args.n_iters]
        n_iters = mid.size
        iters = np.arange(n_iters)
        if not exp_name:
            continue
        color = conf.color(exp_name)
        plt.plot(iters, mid, label=conf.label(exp_name), color=color, linewidth=linewidth)
        plt.fill_between(iters, low, high, alpha=0.25, facecolor=color)
    if n_curves == 0:
        print('Nothing to plot.')
        return 0
    if not args.style:
        plt.xlabel('Iteration', fontsize=fontsize)
        plt.ylabel(args.attr, fontsize=fontsize)
    legend = plt.legend(loc=args.legend_loc, fontsize=fontsize, frameon=False)
    plt.autoscale(enable=True, tight=True)
    plt.tight_layout()
    plt.ylim(args.y_lower, args.y_higher)
    plt.grid(linestyle='--', linewidth='0.2')
    for line in legend.get_lines():
        line.set_linewidth(6.0)
    if not args.output:
        args.output = '{}.pdf'.format(args.attr)
    output_path = os.path.join(args.dir, args.output)
    plt.savefig(output_path)
    plt.clf()


if __name__ == "__main__":
    main()

import os
import pickle
from glob import glob
from shutil import rmtree
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap

from plot.theme import Font, Uniform_CMAP, Sequential_CMAP


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--result_dir", type=str, default=os.path.join(os.getcwd(), "result"))
    parser.add_argument("--figure_dir", type=str, default=os.path.join(os.getcwd(), "figure", "attn"))
    parser.add_argument("--pdf", action="store_true", help="Export PDF")
    args = parser.parse_args()
    return args


def plot_attention(args, attn_path):
    output_dir = os.path.join(args.figure_dir, os.path.normpath(attn_path).split(os.sep)[-3])
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.makedirs(output_dir)

    attention = pickle.load(open(attn_path, "rb"))
    attention = MinMaxScaler().fit_transform(attention)
    for theme_idx, theme in enumerate(Uniform_CMAP + Sequential_CMAP):
        plt.rc("font", **Font)
        fig, ax = plt.subplots()

        cmap = ListedColormap(colormaps[theme].resampled(256)(np.linspace(0.0, 1.0, 256)))
        im = ax.imshow(
            X=attention
            , cmap=cmap
            , vmin=0.0
            , vmax=1.0
        )
        ax.set_xticks([0, 1, 393, 865, 881, 1040])
        ax.set_yticks([0, 1, 393, 865, 881, 1040])
        ax.axes.xaxis.set_ticklabels([None, None, "Video", None, "Language", None])
        ax.axes.yaxis.set_ticklabels([None, None, "Video", None, "Language", None], rotation="vertical", verticalalignment="center")
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
        ax.grid(axis="both", linestyle=(0, (5, 10)))
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_ticks([])
        fig.tight_layout()

        plt.savefig(os.path.join(output_dir, f"{theme_idx}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(output_dir, f"{theme_idx}.pdf"))
        plt.close()


def main():
    args = get_args()
    
    figure_dir = args.figure_dir
    for attn_path in list(glob(os.path.join(args.result_dir, "modal=title_comment_visual_acoustic", "*", "*", "*", "*", "*", "*", "seed=*", "*", "attentions.pkl"))):
        if os.path.exists(os.path.join(os.path.dirname(attn_path), "pretrain")):
            args.figure_dir = os.path.join(figure_dir, "pretrain")
        else:
            args.figure_dir = os.path.join(figure_dir, "train")
        plot_attention(args, attn_path)


if __name__ == "__main__":
    main()
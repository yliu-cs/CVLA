import os
import pickle
from glob import glob
from shutil import rmtree
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from plot.theme import Font, Theme


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--result_dir", type=str, default=os.path.join(os.getcwd(), "result"))
    parser.add_argument("--figure_dir", type=str, default=os.path.join(os.getcwd(), "figure", "tsne"))
    parser.add_argument("--pdf", action="store_true", help="Export PDF")
    args = parser.parse_args()
    return args


def plot_tsne(args, repr_path):
    output_dir = os.path.join(args.figure_dir, os.path.normpath(repr_path).split(os.sep)[-3])
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.makedirs(output_dir)

    representations = pickle.load(open(repr_path, "rb"))
    x = TSNE(
        n_components=2
        , random_state=int(os.path.normpath(repr_path).split(os.sep)[-3].split("=")[-1])
    ).fit_transform(representations)
    references = pickle.load(open(os.path.join(os.path.dirname(repr_path), "predictions.pkl"), "rb"))["references"]
    num_labels = len(np.unique(references))

    for theme_idx, theme in enumerate(list(filter(lambda x: len(x) == num_labels, Theme))):
        plt.rc("font", **Font)
        fig, ax = plt.subplots()

        Marker = ["1", "+"]
        for label in list(reversed(range(num_labels))):
            indices = [i for i, x in enumerate(references) if x == label]
            ax.scatter(
                np.take(x, indices, axis=0)[:, 0]
                , np.take(x, indices, axis=0)[:, 1]
                , color=f"#{theme[label]}"
                , marker=Marker[label]
                , label="Humor" if label == 1 else "Non-Humor"
            )
        ax.legend(prop={ "size": 10 })

        plt.savefig(os.path.join(output_dir, f"{theme_idx}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(output_dir, f"{theme_idx}.pdf"))
        plt.close()


def main():
    args = get_args()
    
    figure_dir = args.figure_dir
    for repr_path in list(glob(os.path.join(args.result_dir, "modal=title_comment_visual_acoustic", "*", "*", "*", "*", "*", "*", "seed=*", "*", "representations.pkl"))):
        if os.path.exists(os.path.join(os.path.dirname(repr_path), "pretrain")):
            args.figure_dir = os.path.join(figure_dir, "pretrain")
        else:
            args.figure_dir = os.path.join(figure_dir, "train")
        plot_tsne(args, repr_path)


if __name__ == "__main__":
    main()
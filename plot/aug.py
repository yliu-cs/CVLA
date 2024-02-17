import os
from shutil import rmtree
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from plot.theme import Sequential_Theme, Projects_Theme, Font


AUG_ACC = np.array([
    [80.290, 80.354, 79.549]
    , [78.744, 80.097, 78.712]
    , [80.676, 79.775, 78.132]
])
X_AXIS = [0.3, 0.5, 0.7]
Y_AXIS = [1, 2, 3]


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--figure_dir", type=str, default=os.path.join(os.getcwd(), "figure"))
    parser.add_argument("--pdf", action="store_true", help="Export PDF")
    args = parser.parse_args()
    return args


def plot_aug_heat(args):
    output_dir = os.path.join(args.figure_dir, "aug")
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.makedirs(output_dir)

    for theme_idx, theme in enumerate(Sequential_Theme + Projects_Theme):
        plt.rc("font", **Font)
        fig, ax = plt.subplots()

        cmap = LinearSegmentedColormap.from_list("", theme)
        bounds = sorted([80., AUG_ACC.min(), AUG_ACC.max()])
        norm = TwoSlopeNorm(vcenter=bounds[1], vmin=bounds[0], vmax=bounds[2])
        im = ax.imshow(
            X=AUG_ACC
            , cmap=cmap
            , norm=norm
        )
        ax.set_xlabel("Apply probability")
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("Number of augmentation")
        ax.yaxis.set_label_position("left")
        ax.set_xticks(np.arange(len(X_AXIS)), labels=X_AXIS)
        ax.set_yticks(np.arange(len(Y_AXIS)), labels=Y_AXIS)
        colors = im.cmap(im.norm(im.get_array()))
        for i in range(AUG_ACC.shape[0]):
            for j in range(AUG_ACC.shape[1]):
                gray = int(int(colors[i][j][0] * 255) * 0.299 + int(colors[i][j][1] * 255) * 0.587 + int(colors[i][j][0] * 255) * 0.114)
                ax.text(j, i, f"{AUG_ACC[i, j]:.3f}" if AUG_ACC[i, j] != 0.0 else 0.0, ha="center", va="center", color="#000000" if gray > 99 else "#FFFFFF")
        ax.spines[:].set_visible(False)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
        ax.set_xticks(np.arange(AUG_ACC.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(AUG_ACC.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        plt.savefig(os.path.join(output_dir, f"{theme_idx}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(output_dir, f"{theme_idx}.pdf"))
        plt.close()


def main():
    args = get_args()

    plot_aug_heat(args)


if __name__ == "__main__":
    main()
import os
import math
import json
from shutil import rmtree
from collections import Counter
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from matplotlib import ticker
import matplotlib.pyplot as plt

from plot.theme import Theme, Font


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(os.getcwd(), "dataset"))
    parser.add_argument("--figure_dir", type=str, default=os.path.join(os.getcwd(), "figure"))
    parser.add_argument("--pdf", action="store_true", help="Export PDF")
    args = parser.parse_args()
    args.figure_dir = os.path.join(args.figure_dir, "dataset", "like")
    return args


def get_like_number(info_path):
    return json.loads(open(info_path, "r", encoding="utf8").read())["like_number"]


def plot_like(args):
    unlabeled_video_dir = os.path.join(args.dataset_dir, "unlabeled")
    unlabeled_likes = []
    for vdir in tqdm(os.listdir(unlabeled_video_dir), desc=f"Like from {os.path.normpath(unlabeled_video_dir).split(os.sep)[-1]}", ncols=100):
        like = get_like_number(os.path.join(unlabeled_video_dir, vdir, "info.json"))
        unlabeled_likes.append(like)
    unlabeled_likes_cnt = Counter(list(map(len, map(str, unlabeled_likes))))
    unlabeled_likes = [unlabeled_likes_cnt[i] for i in range(8)]
    unlabeled_likes = [sum(unlabeled_likes[:5])] + unlabeled_likes[5:]
    
    labeled_video_dir = os.path.join(args.dataset_dir, "labeled")
    labeled_likes = {"pos": [], "neg": []}
    for vdir in tqdm(os.listdir(labeled_video_dir), desc=f"Like from {os.path.normpath(labeled_video_dir).split(os.sep)[-1]}", ncols=100):
        like = get_like_number(os.path.join(labeled_video_dir, vdir, "info.json"))
        if json.loads(open(os.path.join(labeled_video_dir, vdir, "info.json"), "r", encoding="utf8").read())["humor"]:
            labeled_likes["pos"].append(like)
        else:
            labeled_likes["neg"].append(like)
    labeled_likes_cnt = {}
    for lbl in labeled_likes.keys():
        labeled_likes_cnt = Counter(list(map(len, map(str, labeled_likes[lbl]))))
        labeled_likes[lbl] = [labeled_likes_cnt[i] for i in range(8)]
        labeled_likes[lbl] = [sum(labeled_likes[lbl][:5])] + labeled_likes[lbl][5:]

    output_dir = os.path.join(args.figure_dir)
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.makedirs(output_dir)
    
    for theme_idx, theme in enumerate(list(filter(lambda x: len(x) == 3, Theme))):
        plt.rc("font", **Font)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        width = 0.3
        ax1.bar(
            list(map(lambda x: x + 0.5 - width / 2, list(range(4))))
            , unlabeled_likes
            , width=width
            , color="#F9F8F4"
            , edgecolor=f"#{theme[0]}"
            , hatch="xxx"
            , label="Unlabeled"
            , zorder=2
        )
        ax1.set_xlabel("Number of likes")
        ax1.set_ylabel("Number of unlabeled data")
        ax1.set_xlim(left=0, right=4)
        ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "500" if x == 0 else (f"{int(10 ** x)}K" if 1 <= x <= 2 else f"{int(10 ** (x - 3))}M")))
        # ax1.set_xticks(list(range(5)), labels=["0.5k"] + [f"{10 ** (i + 4)}{}" for i in range(1, 5)])
        y_max = max(math.ceil(np.max(unlabeled_likes) / 1000), math.ceil(np.max(np.array(labeled_likes['pos']) + np.array(labeled_likes['neg'])) / 100)) * 1000
        ax1.set_ylim(bottom=0, top=y_max)
        ax1.yaxis.set_major_locator(plt.MultipleLocator(1000))
        ax1.grid(
            axis="y"
            , linestyle=(0, (5, 10))
            , linewidth=0.25
            , color="#4E616C"
            , zorder=0
        )
        for spine in ["top"]:
            ax1.spines[spine].set_color("none")
        
        ax2.bar(
            list(map(lambda x: x + 0.5 + width / 2, list(range(4))))
            , labeled_likes["pos"]
            , width=width
            , color="#F9F8F4"
            , edgecolor=f"#{theme[1]}"
            , hatch="///"
            , label="Humor"
            , zorder=2
        )
        ax2.bar(
            list(map(lambda x: x + 0.5 + width / 2, list(range(4))))
            , labeled_likes["neg"]
            , bottom=labeled_likes["pos"]
            , width=width
            , color="#F9F8F4"
            , edgecolor=f"#{theme[2]}"
            , hatch="\\\\\\"
            , label="Non-Humor"
            , zorder=2
        )
        ax2.set_ylabel("Number of labeled data")
        y_max = int(y_max // 10)
        # y_max = (math.ceil(np.max(np.array(labeled_likes['pos']) + np.array(labeled_likes['neg'])) / 100) + 1) * 100
        ax2.set_ylim(bottom=0, top=y_max)
        ax2.yaxis.set_major_locator(plt.MultipleLocator(100))
        for spine in ["top"]:
            ax2.spines[spine].set_color("none")
        
        ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
        ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
        ax1.legend(
            ax1_handles + ax2_handles
            , ax1_labels + ax2_labels
            , loc="upper right"
            , prop={
                "size": 10
            }
        )

        plt.savefig(os.path.join(output_dir, f"{theme_idx}.png"), dpi=600)
        if args.pdf:
            plt.savefig(os.path.join(output_dir, f"{theme_idx}.pdf"))
        plt.close()


def main():
    args = get_args()

    plot_like(args)


if __name__ == "__main__":
    main()
import os
import math
import json
from shutil import rmtree
from collections import Counter
from argparse import ArgumentParser

import av
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from plot.theme import Theme, Font


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(os.getcwd(), "dataset"))
    parser.add_argument("--figure_dir", type=str, default=os.path.join(os.getcwd(), "figure"))
    parser.add_argument("--pdf", action="store_true", help="Export PDF")
    args = parser.parse_args()
    args.figure_dir = os.path.join(args.figure_dir, "dataset", "duration")
    return args


def get_video_duration(video_path):
    container = av.open(video_path)
    video = container.streams.video[0]
    return float(video.duration * video.time_base)


def plot_duration(args):
    unlabeled_video_dir = os.path.join(args.dataset_dir, "unlabeled")
    unlabeled_durations = []
    for vdir in tqdm(os.listdir(unlabeled_video_dir), desc=f"Duration from {os.path.normpath(unlabeled_video_dir).split(os.sep)[-1]}", ncols=100):
        duration = get_video_duration(os.path.join(unlabeled_video_dir, vdir, "video.mp4"))
        unlabeled_durations.append(duration)
    print(f"Sum duration of unlabeled videos {round(sum(unlabeled_durations) / 60 ** 2, 2)} hours.")
    unlabeled_durations_cnt = Counter(list(map(lambda x: int(x // 10), unlabeled_durations)))
    unlabeled_durations = [unlabeled_durations_cnt[i] for i in range(6)]
    unlabeled_durations = unlabeled_durations[:3] + [sum(unlabeled_durations[3:])]

    labeled_video_dir = os.path.join(args.dataset_dir, "labeled")
    labeled_durations = {"pos": [], "neg": []}
    for vdir in tqdm(os.listdir(labeled_video_dir), desc=f"Duration from {os.path.normpath(labeled_video_dir).split(os.sep)[-1]}", ncols=100):
        duration = get_video_duration(os.path.join(labeled_video_dir, vdir, "video.mp4"))
        if json.loads(open(os.path.join(labeled_video_dir, vdir, "info.json"), "r", encoding="utf8").read())["humor"]:
            labeled_durations["pos"].append(duration)
        else:
            labeled_durations["neg"].append(duration)
    print(f"Sum duration of labeled videos {round(sum(labeled_durations['pos']+labeled_durations['neg']) / 60 ** 2, 2)} hours.")
    labeled_durations_cnt = {}
    for lbl in labeled_durations.keys():
        labeled_durations_cnt[lbl] = Counter(list(map(lambda x: int(x // 10), labeled_durations[lbl])))
        labeled_durations[lbl] = [labeled_durations_cnt[lbl][i] for i in range(6)]
        labeled_durations[lbl] = labeled_durations[lbl][:3] + [sum(labeled_durations[lbl][3:])]

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
            , unlabeled_durations
            , width=width
            , color="#F9F8F4"
            , edgecolor=f"#{theme[0]}"
            , hatch="xxx"
            , label="Unlabeled"
            , zorder=2
        )
        ax1.set_xlabel("Duration of videos")
        ax1.set_ylabel("Number of unlabeled data")
        ax1.set_xlim(left=0, right=4)
        ax1.set_xticks(list(range(5)), labels=["5s"] + [f"{i * 10}s" for i in range(1, 4)] + ["60s"])
        y_max = max(math.ceil(np.max(unlabeled_durations) / 1000), math.ceil(np.max(np.array(labeled_durations['pos']) + np.array(labeled_durations['neg'])) / 100)) * 1000
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
            , labeled_durations["pos"]
            , width=width
            , color="#F9F8F4"
            , edgecolor=f"#{theme[1]}"
            , hatch="///"
            , label="Humor"
            , zorder=2
        )
        ax2.bar(
            list(map(lambda x: x + 0.5 + width / 2, list(range(4))))
            , labeled_durations["neg"]
            , bottom=labeled_durations["pos"]
            , width=width
            , color="#F9F8F4"
            , edgecolor=f"#{theme[2]}"
            , hatch="\\\\\\"
            , label="Non-Humor"
            , zorder=2
        )
        ax2.set_ylabel("Number of labeled data")
        y_max = int(y_max // 10)
        # y_max = (math.ceil(np.max(np.array(labeled_durations['pos']) + np.array(labeled_durations['neg'])) / 100) + 1) * 100
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

    plot_duration(args)


if __name__ == "__main__":
    main()
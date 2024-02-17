import os
from glob import glob
from shutil import rmtree
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

from plot.theme import Theme, Font


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--result_dir", type=str, default=os.path.join(os.getcwd(), "result"))
    parser.add_argument("--figure_dir", type=str, default=os.path.join(os.getcwd(), "figure"))
    parser.add_argument("--modal", type=str, default="text_video")
    parser.add_argument("--pdf", action="store_true", help="Export PDF")
    args = parser.parse_args()
    args.modal = args.modal.replace("video", "visual_acoustic").replace("text", "title_comment")
    return args


def load_loss(args):
    pretrain_loss, finetune_acc, train_acc = [], [], []
    log_pattern_path = glob(os.path.join(args.result_dir, f"modal={args.modal}", "*", "*", "*", "*", "*", "*", "seed=*", "*", "run.log"))
    for log_path in sorted(log_pattern_path, key=lambda x: int(os.path.normpath(x).split(os.sep)[-3].split("=")[-1])):
        loss, acc, flag = [], [], False
        for line in open(log_path, "r").readlines():
            line = line.split(" [INFO]: ")[-1].strip()
            if "Pre-Training" in line:
                flag = True
            if line.startswith("Pre-Training of Epoch "):
                loss.append(float(line.split(" ")[-1].split("=")[-1]))
            elif line.startswith("Training of Epoch "):
                acc.append(float(line.split("Dev Metrics: ")[-1].split(" ")[1].split("=")[-1]))
        if loss:
            pretrain_loss.append(loss)
        if flag:
            finetune_acc.append(acc)
        else:
            train_acc.append(acc)
    return tuple(map(np.array, [pretrain_loss, finetune_acc, train_acc]))


def plot_loss_curve(args):
    pretrain_loss, finetune_acc, train_acc = load_loss(args)

    output_dir = os.path.join(args.figure_dir, "loss_curve")
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.makedirs(output_dir)

    for theme_idx, theme in enumerate(list(filter(lambda x: len(x) == 3, Theme))):
        plt.rc("font", **Font)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(
            range(1, pretrain_loss.shape[1] + 1)
            , pretrain_loss[1:-1, :].mean(axis=0)
            , color=f"#{theme[0]}"
            , marker="o"
            , linestyle="-"
            , label="Loss of PT"
        )
        ax1.fill_between(
            range(1, pretrain_loss.shape[1] + 1)
            , pretrain_loss[1:-1, :].mean(axis=0) - pretrain_loss[1:-1, :].std(axis=0)
            , pretrain_loss[1:-1, :].mean(axis=0) + pretrain_loss[1:-1, :].std(axis=0)
            , color=f"#{theme[0]}"
            , alpha=0.1
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_xlim(left=1, right=max(pretrain_loss.shape[1], finetune_acc.shape[1], train_acc.shape[1]))
        # ax1.set_xlim(left=1, right=pretrain_loss.shape[1])
        ax1.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax1.yaxis.set_major_locator(plt.MultipleLocator(4))
        for spine in ["top"]:
            ax1.spines[spine].set_color("none")

        ax2.plot(
            range(1, finetune_acc.shape[1] + 1)
            , finetune_acc[1:-1, :].mean(axis=0)
            , color=f"#{theme[1]}"
            , marker="s"
            , linestyle="-."
            , label="Dev Acc of HD w/ PT"
        )
        ax2.fill_between(
            range(1, finetune_acc.shape[1] + 1)
            , finetune_acc[1:-1, :].mean(axis=0) - finetune_acc[1:-1, :].std(axis=0)
            , finetune_acc[1:-1, :].mean(axis=0) + finetune_acc[1:-1, :].std(axis=0)
            , color=f"#{theme[1]}"
            , alpha=0.1
        )
        ax2.plot(
            range(1, train_acc.shape[1] + 1)
            , train_acc[1:-1, :].mean(axis=0)
            , color=f"#{theme[2]}"
            , marker="v"
            , linestyle="--"
            , label="Dev Acc of HD w/o PT"
        )
        ax2.fill_between(
            range(1, train_acc.shape[1] + 1)
            , train_acc[1:-1, :].mean(axis=0) - train_acc[1:-1, :].std(axis=0)
            , train_acc[1:-1, :].mean(axis=0) + train_acc[1:-1, :].std(axis=0)
            , color=f"#{theme[2]}"
            , alpha=0.1
        )
        ax2.set_ylabel("Accuracy")
        ax2.yaxis.set_major_locator(plt.MultipleLocator(10))
        for spine in ["top"]:
            ax2.spines[spine].set_color("none")
        
        ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
        ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
        ax1.legend(
            ax1_handles + ax2_handles
            , ax1_labels + ax2_labels
            , loc="lower right"
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

    plot_loss_curve(args)


if __name__ == "__main__":
    main()
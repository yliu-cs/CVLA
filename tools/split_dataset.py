import os
import sys
import json
import pickle
from collections import Counter
from argparse import ArgumentParser

from transformers import set_seed
from sklearn.model_selection import train_test_split


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(os.getcwd(), "dataset", "labeled"))
    parser.add_argument("--seed", nargs="+", type=int, default=[2, 42, 327, 2023, 998244353])
    parser.add_argument("--train_size", type=int, default=100)
    args = parser.parse_args()
    args.output_dir = os.path.join(args.dataset_dir, os.pardir, "split", "train")
    return args


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
    vids = list(os.listdir(args.dataset_dir))
    labels = list(map(lambda x: json.loads(open(os.path.join(args.dataset_dir, x, "info.json"), "r", encoding="utf8").read())["humor"], vids))
    for seed in args.seed:
        set_seed(seed)
        train_vids, test_vids, train_labels, test_labels = train_test_split(vids, labels, train_size=args.train_size, random_state=seed)
        dev_vids, test_vids, dev_labels, test_labels = train_test_split(test_vids, test_labels, train_size=args.train_size, random_state=seed)
        print(f"[+] {seed=}")
        print(f"  Train split: {Counter(train_labels)}")
        print(f"  Dev split: {Counter(dev_labels)}")
        print(f"  Test split: {Counter(test_labels)}")
        pickle.dump({"train": train_vids, "dev": dev_vids, "test": test_vids}, open(os.path.join(args.output_dir, f"{seed=}.pkl"), "wb"))


if __name__ == "__main__":
    main()
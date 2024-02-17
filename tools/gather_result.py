import os
from argparse import ArgumentParser

import numpy as np


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--result_dir", type=str, default=os.path.join(os.getcwd(), "result"), help="Path to result")
    parser.add_argument("--mid", action="store_false", help="Drop the maximum and minimum values")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seed")
    parser.add_argument("--require", nargs="+", type=str, default=None, help="Filter")
    parser.add_argument("--exclude", nargs="+", type=str, default=None, help="Filter")
    args = parser.parse_args()
    return args


def format_dir(dir):
    return ", ".join(list(os.path.normpath(dir).split(os.sep))[list(os.path.normpath(dir).split(os.sep)).index("result") + 1:])


def format_result(result):
    result_string = ""
    for mode in result.keys():
        result_string += f"  + {mode:>5}: "
        result_string += ", ".join([f"{k}={v['avg']:.3f}±{v['std']:.3f}" for k, v in result[mode].items()])
        result_string += "\n"
    return result_string


def gather_result(args, cur_dir):
    if "run.log" in os.listdir(cur_dir):
        result = {}
        with open(os.path.join(cur_dir, "run.log"), "r") as log_file:
            for line in log_file.readlines():
                line = line.split(" [INFO]: ")[-1].strip()
                if line.startswith(("[*]", "[#]")):
                    mode = line.split(": ")[0].split(" ")[-2]
                    if mode not in result:
                        result[mode] = {}
                    for string_result in line.split(": ")[-1].split(" "):
                        key, value = string_result.split("=")
                        result[mode][key] = float(value)
        return result

    if sum(list(map(lambda x: x.startswith("seed="), os.listdir(cur_dir)))) == args.num_seeds:
        result = {}
        for sub_dir in os.listdir(cur_dir):
            if not os.path.isdir(os.path.join(cur_dir, sub_dir)):
                continue
            sub_result = gather_result(args, os.path.join(cur_dir, sub_dir))
            for mode in sub_result.keys():
                if mode not in result:
                    result[mode] = {}
                for k in sub_result[mode].keys():
                    if k not in result[mode]:
                        result[mode][k] = []
                    result[mode][k].append(sub_result[mode][k])
        for mode in result.keys():
            if args.mid:
                min_idx = result[mode]["Acc"].index(min(result[mode]["Acc"]))
                max_idx = result[mode]["Acc"].index(max(result[mode]["Acc"]))
                for k in result[mode].keys():
                    del result[mode][k][max(min_idx, max_idx)]
                    del result[mode][k][min(min_idx, max_idx)]
                for k in result[mode].keys():
                    result[mode][k] = {
                        "avg": round(np.mean(result[mode][k]), 3)
                        , "std": round(np.std(result[mode][k]), 3)
                    }
        result_string = format_result(result)
        if (args.require is None or sum([require in format_dir(cur_dir) for require in args.require]) == len(args.require)) \
            and (args.exclude is None or not sum([exclude in format_dir(cur_dir) for exclude in args.exclude])):
            print(f"[#] {format_dir(cur_dir)}\n{result_string}")
    elif os.path.normpath(cur_dir).split(os.sep)[-1].startswith("seed="):
        assert len(os.listdir(cur_dir)) == 1
        if os.path.isdir(os.path.join(cur_dir, os.listdir(cur_dir)[0])):
            return gather_result(args, os.path.join(cur_dir, os.listdir(cur_dir)[0]))
    else:
        sub_dir_list = os.listdir(cur_dir)
        if sum(map(lambda x: x.startswith("modal="), sub_dir_list)):
            sub_dir_list.sort(key=lambda x: len(x.split("=")[-1].split("_")))
        for sub_dir in sub_dir_list:
            if os.path.isdir(os.path.join(cur_dir, sub_dir)):
                gather_result(args, os.path.join(cur_dir, sub_dir))


def main():
    args = get_args()

    gather_result(args, args.result_dir)


if __name__ == "__main__":
    main()

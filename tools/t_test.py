from argparse import ArgumentParser

import numpy as np
from scipy.stats import ttest_ind_from_stats


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--src_result", type=str, required=True)
    parser.add_argument("--dst_result", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()
    return args


def t_test(x, y, num_samples):
    def modify_std(std):
        return np.sqrt(np.float32(num_samples) / np.float32(num_samples - 1)) * std
    statistic, p_value = ttest_ind_from_stats(x[0], modify_std(x[1]), num_samples, y[0], modify_std(y[1]), num_samples)
    return p_value


def main():
    args = get_args()

    string_result = []
    for _x, _y in zip(args.src_result.split(", "), args.dst_result.split(", ")):
        assert _x.split("=")[0] == _y.split("=")[0]
        x, y = [list(map(float, _.split("=")[-1].split("±"))) for _ in [_x, _y]]
        p_value = t_test(x, y, args.num_samples)
        string_result.append(f"{_x.split('=')[0]}={p_value:.3f}")
    print("[+] \"P value\": " + ", ".join(string_result))


if __name__ == "__main__":
    main()
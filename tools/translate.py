import os
import json
import pickle
from argparse import ArgumentParser

import translators
from tqdm import tqdm


translators.preaccelerate()


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.getcwd(), "dataset", "labeled"))
    parser.add_argument("--store_path", type=str, default=os.path.join(os.getcwd(), "dataset", "vid2en.pkl"))
    args = parser.parse_args()
    return args


def text_preprocess(text):
    text = text.replace("作者赞过", "").replace("作者回复过", "").replace("置顶", "").replace("#搞笑", "").replace("#搞笑视频", "")
    return text.strip()


def translate(text):
    return translators.translate_text(text, translator="baidu", from_language="zh", to_language="en")


def main():
    args = get_args()

    vids = os.listdir(args.data_dir)

    vid2en = {}
    if os.path.exists(args.store_path):
        vid2en = pickle.load(open(args.store_path, "rb"))
    
    for vid in tqdm(vids, desc="Translate", ncols=100):
        if vid in vid2en:
            continue
        vdir = os.path.join(args.data_dir, vid)
        title = text_preprocess(json.loads(open(os.path.join(vdir, "info.json"), "r", encoding="utf8").read())["title"])
        comments = " ".join(list(map(lambda x: text_preprocess(x["content"]), json.loads(open(os.path.join(vdir, "comment.json"), "r", encoding="utf8").read()))))
        try:
            vid2en[vid] = translate(title + comments)
            pickle.dump(vid2en, open(args.store_path, "wb"))
        except Exception as ex:
            print(f"{ex=}")
    
    print(f"Translate labeled data {len(vid2en)}/{len(vids)}.")


if __name__ == "__main__":
    main()
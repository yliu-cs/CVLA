import os
import subprocess
from shutil import copyfile
from argparse import ArgumentParser

from tqdm import tqdm


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--src_dataset_dir", type=str, default=os.path.join(os.getcwd(), os.pardir, "VHD", "dataset"))
    parser.add_argument("--dst_dataset_dir", type=str, default=os.path.join(os.getcwd(), "dataset"))
    args = parser.parse_args()
    return args


def transfer_data(src_data_dir, dst_data_dir):
    for vid in tqdm(os.listdir(src_data_dir), desc=f"{os.path.normpath(src_data_dir).split(os.sep)[-1]}", ncols=100):
        src_vdir = os.path.join(src_data_dir, vid)
        dst_vdir = os.path.join(dst_data_dir, vid)
        if not os.path.exists(dst_vdir):
            os.makedirs(dst_vdir)
        if not os.path.exists(os.path.join(dst_vdir, "video.mp4")):
            copyfile(os.path.join(src_vdir, "video.mp4"), os.path.join(dst_vdir, "video.mp4"))
        if not os.path.exists(os.path.join(dst_vdir, "info.json")):
            copyfile(os.path.join(src_vdir, "info.json"), os.path.join(dst_vdir, "info.json"))
        if not os.path.exists(os.path.join(dst_vdir, "comment.json")):
            copyfile(os.path.join(src_vdir, "comment.json"), os.path.join(dst_vdir, "comment.json"))


def transfer_dataset(src_dataset_dir, dst_dataset_dir):
    for tag in ["labeled", "unlabeled"]:
        src_data_dir = os.path.join(src_dataset_dir, tag)
        dst_data_dir = os.path.join(dst_dataset_dir, tag)
        if tag == "labeled":
            for mode in ["train", "val", "test"]:
                transfer_data(os.path.join(src_data_dir, mode), dst_data_dir)
        elif tag == "unlabeled":
                transfer_data(src_data_dir, dst_data_dir)
        else:
            raise NotImplementedError


def ffmpeg_convert_mp4_to_wav(video_path, audio_path):
    subprocess.call(["ffmpeg", "-y", "-i", video_path, audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def convert_video_to_audio(dataset_dir):
    for tag in ["labeled", "unlabeled"]:
        for vid in tqdm(os.listdir(os.path.join(dataset_dir, tag)), desc=f"convert {tag} wav", ncols=100):
            vdir = os.path.join(dataset_dir, tag, vid)
            if not os.path.exists(os.path.join(vdir, "audio.wav")):
                ffmpeg_convert_mp4_to_wav(os.path.join(vdir, "video.mp4"), os.path.join(vdir, "audio.wav"))


def main():
    args = get_args()

    transfer_dataset(args.src_dataset_dir, args.dst_dataset_dir)
    convert_video_to_audio(args.dst_dataset_dir)


if __name__ == "__main__":
    main()
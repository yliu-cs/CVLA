import os
from shutil import rmtree, copyfile
from argparse import ArgumentParser

import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

from plot.theme import Font


def get_args():
    data_dir = os.path.join(os.getcwd(), "dataset", "labeled")
    parser = ArgumentParser()
    parser.add_argument("--vid", type=str, default=os.listdir(data_dir)[0])
    parser.add_argument("--figure_dir", type=str, default=os.path.join(os.getcwd(), "figure"))
    parser.add_argument("--pdf", action="store_true", help="Export PDF")
    args = parser.parse_args()
    args.audio_path = os.path.join(data_dir, args.vid, "audio.wav")
    return args


def plot_audio_mel(args):
    y, sr = librosa.load(args.audio_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    output_dir = os.path.join(args.figure_dir, "mel", os.path.normpath(args.audio_path).split(os.sep)[-2])
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.makedirs(output_dir)

    plt.rc("font", **Font)
    fig, ax = plt.subplots()

    img = librosa.display.specshow(
        mel_db
        , x_axis="time"
        , y_axis="mel"
        , sr=sr
        , fmax=8000
        , ax=ax
    )
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    # ax.set_title("Mel-frequency spectrogram")
    
    copyfile(args.audio_path, os.path.join(output_dir, os.path.basename(args.audio_path)))
    plt.savefig(os.path.join(output_dir, "mel.png"), dpi=600)
    if args.pdf:
        plt.savefig(os.path.join(output_dir, "mel.pdf"))
    plt.close()


def main():
    args = get_args()

    plot_audio_mel(args)


if __name__ == "__main__":
    main()
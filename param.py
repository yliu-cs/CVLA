import os
import sys
import logging
from ast import literal_eval
from datetime import datetime

import torch


cuda = 0
seed = 42
dataset_dir = os.path.join(os.getcwd(), "dataset")
output_dir = os.path.join(os.getcwd(), "result")

modal = "text_video"
frame_sample_rate = 4
samplerate = 8000
hop_len = 512
n_mels = 128
num_comments = 10
text_max_len = 16
visual_patch_size = (6, 32, 32)
acoustic_patch_size = (16, 16)
num_labels = 2

max_duration = 60
clip_len = 10 # second
num_clip_frames = 8
text_encoder = "bert-base-chinese"
embedding_dim = 768
video_hidden_dim = 1024
video_nhead = 12
video_num_layers = 8
fusion_hidden_dim = 1024
fusion_nhead = 12
fusion_num_layers = 8
dropout = 0.1

do_train = True
do_pretrain = True
pretrain_scale = None
pretrain_epoch = 10
pretrain_shuffle = True
pretrain_lr = 1e-5
pretrain_weight_decay = 1e-2
pretrain_visual_aug_p = 0.3
pretrain_num_visual_aug = 3
epoch = 20
batch_size = 6
train_shuffle = True
optim = "AdamW"
lr = 1e-5
weight_decay = 1e-2
dev_metric = "acc"


logger = logging.getLogger(__name__)


def load_default_params():
    params = {
        k: v for k, v in globals().items() \
            if not k.startswith("_") \
                and isinstance(v, (int, float, bool, str, tuple, type(None))) \
                and k != "logger"
    }
    return params


def overriding_params(params):
    for arg in sys.argv[1:]:
        if "=" not in arg:
            assert not arg.startswith("--")
            config_file = f"{arg}.py"
            logger.info(f"Overriding Config Wih {config_file}:")
            with open(config_file) as f:
                logger.info(f.read())
            exec(open(config_file).read())
        else:
            assert arg.startswith("--")
            key, val = arg.split("=")
            key = key[2:]
            if key in params.keys():
                try:
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    attempt = val
                assert (type(attempt) == type(params[key])) or (attempt == None) or (params[key] == None)
                logger.info(f"Overriding: {key} = {attempt}")
                params[key] = attempt
            else:
                raise ValueError(f"Unknow config key: {key}")
    return params


def complete_params(params):
    if params["cuda"] < torch.cuda.device_count():
        params["gpu"] = torch.cuda.get_device_name(params["cuda"])
    else:
        raise ValueError(f"Not available GPU: cuda:{params['cuda']}.")
    
    params["modal"] = params["modal"].replace("video", "visual_acoustic").replace("text", "title_comment")
    params["max_num_frames"] = int(params["max_duration"] * (params["num_clip_frames"] / params["clip_len"]))
    params["max_temporal_len"] = int(params["max_num_frames"] // visual_patch_size[0])
    params["max_mel_len"] = int((params["max_duration"] * params["samplerate"]) // params["hop_len"])
    params["max_mel_len"] += params["acoustic_patch_size"][0] - (params["max_mel_len"] % params["acoustic_patch_size"][0])
    params["max_acoustic_len"] = int((params["max_mel_len"] // acoustic_patch_size[0]) * (params["n_mels"] // acoustic_patch_size[1]))
    
    params["output_dir"] = os.path.join(
        params["output_dir"]
        , f"modal={params['modal']}"
        , f"pretrain_epoch={params['pretrain_epoch']}&pretrain_lr={params['pretrain_lr']}&pretrain_weight_decay={params['pretrain_weight_decay']}"
        , f"pretrain_visual_aug_p={params['pretrain_visual_aug_p']}&pretrain_num_visual_aug={params['pretrain_num_visual_aug']}&pretrain_scale={params['pretrain_scale']}"
        , f"epoch={params['epoch']}&lr={params['lr']}&weight_decay={params['weight_decay']}"
        , f"visual_patch_size={params['visual_patch_size']}&acoustic_patch_size={params['acoustic_patch_size']}"
        , f"video_hidden_dim={params['video_hidden_dim']}&video_nhead={params['video_nhead']}&video_num_layers={params['video_num_layers']}"
        , f"fusion_hidden_dim={params['fusion_hidden_dim']}&fusion_nhead={params['fusion_nhead']}&fusion_num_layers={params['fusion_num_layers']}"
        , f"seed={params['seed']}", datetime.today().strftime("%Y.%m.%d-%H:%M:%S")
    )
    return params


def show_params(params):
    for k, v in params.items():
        logger.info(f"[H] {k}: {v}")


if __name__ == "__main__":
    params = load_default_params()
    params = overriding_params(params=params)
    params = complete_params(params=params)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s]: %(message)s"
        , datefmt="%Y.%m.%d-%H:%M:%S"
        , level=logging.INFO
        , filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log"
        , filemode="w"
    )
    show_params(params)
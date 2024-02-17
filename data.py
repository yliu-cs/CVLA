import os
import json
import pickle
from random import randint

import av
import torch
import librosa
import torchaudio
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, params, mode="train"):
        super().__init__()
        self.params = params
        self.mode = mode
        self.feature_path = os.path.join(
            params["dataset_dir"], "feature"
            , f"max_num_frames={params['max_num_frames']}"
            , f"max_mel_len={params['max_mel_len']}"
            , f"text_max_len={params['text_max_len']}"
        )
        if not os.path.exists(self.feature_path):
            os.makedirs(self.feature_path)
        self.data_path = os.path.join(params["dataset_dir"], "unlabeled" if mode == "pretrain" else "labeled")
        if mode == "pretrain":
            self.vids = list(os.listdir(self.data_path))
            if params["pretrain_scale"] is not None:
                self.vids = self.vids[:params["pretrain_scale"] + randint(-100, 100)]
        else:
            self.vids = pickle.load(open(os.path.join(params["dataset_dir"], "split", "train", f"seed={params['seed']}.pkl"), "rb"))[mode]
        
        self.text_tokenizer = AutoTokenizer.from_pretrained(params["text_encoder"])
    
    def __len__(self):
        return len(self.vids)

    def __read_visual__(self, visual_path):
        container = av.open(visual_path)
        visual = container.streams.video[0]
        assert visual.frames >= self.params["max_num_frames"]
        indices = np.linspace(0, visual.frames, num=self.params["max_num_frames"], dtype=np.int64)
        frames = np.stack(list(map(lambda x: x.to_ndarray(format="rgb24"), list(container.decode(video=0)))))
        indices = np.clip(indices, 0, frames.shape[0] - 1)
        visual = torch.from_numpy(frames[indices]).float().permute(0, 3, 1, 2)
        if self.mode.endswith("train"):
            visual = transforms.Resize([256, 256])(visual)
            visual = transforms.RandomCrop([224, 224])(visual)
        else:
            visual = transforms.Resize([224, 224])(visual)
        visual = visual.permute(1, 0, 2, 3)
        return visual

    def __read_acoustic__(self, acoustic_path):
        acoustic, raw_samplerate = torchaudio.load(acoustic_path)
        if raw_samplerate != self.params["samplerate"]:
            acoustic = torchaudio.functional.resample(acoustic, orig_freq=raw_samplerate, new_freq=self.params["samplerate"])
        acoustic = acoustic.mean(0).numpy()
        acoustic = acoustic - acoustic.mean()
        acoustic = librosa.feature.melspectrogram(y=acoustic, sr=self.params["samplerate"], hop_length=self.params["hop_len"], n_mels=self.params["n_mels"])
        acoustic = librosa.power_to_db(acoustic) - 20.0
        acoustic = np.clip(acoustic / 40.0, -2.0, 0.0) + 1.0
        acoustic = torch.from_numpy(acoustic)
        assert self.params["acoustic_patch_size"][0] == self.params["acoustic_patch_size"][1]
        if acoustic.size(1) < self.params["max_mel_len"]:
            acoustic = F.pad(acoustic, (0, self.params["max_mel_len"] - acoustic.size(1)), "constant", 0.0)
        acoustic = acoustic.transpose(0, 1).unsqueeze(0)
        return acoustic
    
    def __text_preprocess__(self, text):
        text = text.replace("作者赞过", "").replace("作者回复过", "").replace("置顶", "").replace("#搞笑", "").replace("#搞笑视频", "")
        return text.strip()
    
    def __read_text__(self, text):
        tokens = self.text_tokenizer.tokenize(self.__text_preprocess__(text))
        input_ids = [self.text_tokenizer.cls_token_id] + self.text_tokenizer.convert_tokens_to_ids(tokens) + [self.text_tokenizer.sep_token_id]
        attn_mask = [1 for _ in range(len(input_ids))]
        if len(input_ids) > self.params["text_max_len"]:
            input_ids = input_ids[:self.params["text_max_len"] - 1] + input_ids[-1:]
            attn_mask = attn_mask[:self.params["text_max_len"] - 1] + attn_mask[-1:]
        while len(input_ids) < self.params["text_max_len"]:
            input_ids = input_ids[:-1] + [self.text_tokenizer.pad_token_id] + input_ids[-1:]
            attn_mask = attn_mask[:-1] + [0] + attn_mask[-1:]
        return input_ids, attn_mask
    
    def __get_sample__(self, vid):
        if os.path.exists(os.path.join(self.feature_path, f"{vid}.pkl")):
            return pickle.load(open(os.path.join(self.feature_path, f"{vid}.pkl"), "rb"))
        vdir = os.path.join(self.data_path, vid)
        inputs = ()
        # visual
        visual = self.__read_visual__(os.path.join(vdir, "video.mp4"))
        inputs = inputs + (visual, )
        # acoustic
        acoustic    = self.__read_acoustic__(os.path.join(vdir, "audio.wav"))
        inputs = inputs + (acoustic, )
        # title
        title_input_ids, title_attn_mask = self.__read_text__(json.loads(open(os.path.join(vdir, "info.json"), "r", encoding="utf8").read())["title"])
        title_input_ids, title_attn_mask = [torch.LongTensor(_) for _ in [title_input_ids, title_attn_mask]]
        inputs = inputs + ((title_input_ids, title_attn_mask), )
        # comment
        comment_block_list = json.loads(open(os.path.join(vdir, "comment.json"), "r", encoding="utf8").read())
        comment_input_ids, comment_attn_mask = [], []
        for comment_block in comment_block_list:
            input_ids, attn_mask = self.__read_text__(comment_block["content"])
            comment_input_ids += input_ids
            comment_attn_mask += attn_mask
        if len(comment_block_list) < self.params["num_comments"]:
            for i in range(self.params["num_comments"] - len(comment_block_list)):
                comment_input_ids += [self.text_tokenizer.pad_token_id for _ in range(self.params["text_max_len"])]
                comment_attn_mask += [0 for _ in range(self.params["text_max_len"])]
        for i in range(len(comment_input_ids)):
            if i != 0 and comment_input_ids[i] == self.text_tokenizer.cls_token_id:
                comment_input_ids[i] = self.text_tokenizer.sep_token_id
        comment_input_ids, comment_attn_mask = [torch.LongTensor(_) for _ in [comment_input_ids, comment_attn_mask]]
        inputs = inputs + ((torch.LongTensor(comment_input_ids), torch.LongTensor(comment_attn_mask)), )
        pickle.dump(inputs, open(os.path.join(self.feature_path, f"{vid}.pkl"), "wb"))
        return inputs

    def __getitem__(self, idx):
        vid = self.vids[idx]
        inputs = self.__get_sample__(vid)
        if self.mode == "pretrain":
            return inputs
        else:
            label = torch.LongTensor([1 if json.loads(open(os.path.join(self.data_path, vid, "info.json"), "r", encoding="utf8").read())["humor"] else 0]).squeeze()
            return (inputs, label)
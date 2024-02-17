import os
import pickle
import logging
import warnings
from glob import glob
from itertools import product
from collections.abc import Iterable
from shutil import copyfile, make_archive, rmtree

import torch
import numpy as np
import transformers
from tqdm import tqdm
from torch.cuda import amp
from transformers import set_seed
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from models import CVLA
from data import Dataset
from param import load_default_params, overriding_params, complete_params, show_params


logger = logging.getLogger(__name__)


def ignore_warnings():
    warnings.filterwarnings("ignore")
    transformers.logging.set_verbosity_error()


def calc_model_size(model):
    param_size, param_num = 0, 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_num += param.nelement()
    buffer_size, buffer_num = 0, 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_num += buffer.nelement()
    model_size, model_num = param_size + buffer_size, param_num + buffer_num
    
    def clever_format(nums, format="%.2f"):
        if not isinstance(nums, Iterable):
            nums = [nums]
        clever_nums = []
        for num in nums:
            for scale, suffix in zip([1e12, 1e9, 1e6, 1e3, 1], ["T", "G", "M", "K", "B"]):
                if num >= scale:
                    clever_nums.append(format % (num / scale) + suffix)
                    break
        clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)
        return clever_nums

    model_size, model_num = clever_format([model_size, model_num], format="%.2f")
    return model_size, model_num


def calc_metrics(predictions, references):
    acc = accuracy_score(references, predictions) * 100
    _mac = precision_recall_fscore_support(references, predictions, average="macro")
    _wtd = precision_recall_fscore_support(references, predictions, average="weighted")
    mac, wtd = {}, {}
    for i, x in enumerate(("pre", "rec", "f1")):
        mac[x], wtd[x] = _mac[i] * 100, _wtd[i] * 100
    metrics = {"acc": acc, "mac": mac, "wtd": wtd}
    return metrics


def inline_format_metrics(metrics):
    format_metrics = {"acc".capitalize(): metrics["acc"]}
    for x in product(("mac", "wtd"), ("pre", "rec", "f1")):
        format_metrics[f"{x[0].capitalize()}-{x[1].capitalize()}"] = metrics[x[0]][x[1]]
    string_metrics = " ".join([f"{key}={value:.3f}" for key, value in format_metrics.items()])
    return string_metrics


def save_code(project_path, save_dir):
    save_path = os.path.join(save_dir, os.path.basename(os.path.normpath(project_path)))

    def filter_copy_file(src_path):
        reserve_file = (".py", ".sh", ".md", ".txt")
        if os.path.isfile(src_path):
            if src_path.endswith(reserve_file):
                dst_path = os.path.join(save_path, os.path.relpath(src_path, project_path))
                if not os.path.exists(os.path.dirname(dst_path)):
                    os.makedirs(os.path.dirname(dst_path))
                copyfile(src_path, dst_path)
        elif os.path.isdir(src_path):
            for file in os.listdir(src_path):
                if file != "dataset":
                    filter_copy_file(os.path.join(src_path, file))
        else:
            raise NotImplementedError

    filter_copy_file(project_path)
    make_archive(save_path, "zip", save_path)
    rmtree(save_path)


def move_to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    elif isinstance(x, (tuple, list)):
        return tuple(list(map(move_to_cuda, x)))
    else:
        return x


class PreTrainer(object):
    def __init__(self, params, model, pertrain_dataset):
        self.params = params
        logger.info("Inititalizing Pre-Trainer ...")

        self.pretrain_loader = DataLoader(pertrain_dataset, drop_last=True, batch_size=self.params["batch_size"], shuffle=self.params["pretrain_shuffle"])

        self.model = model.cuda()
        self.optimizer = getattr(torch.optim, self.params["optim"])(
            filter(lambda p: p.requires_grad, self.model.parameters())
            , lr=self.params["pretrain_lr"]
            , weight_decay=self.params["pretrain_weight_decay"]
        )
        self.scaler = amp.GradScaler()

        model_size, model_num = calc_model_size(self.model)
        logger.info(f"[-] Number of Model Parameters={model_num} & Model Size={model_size}.")
    
    def save_best_model(self):
        output_dir = os.path.join(self.params["output_dir"], "pretrain")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        logger.info(f"  [|] Saving Best Model to {self.params['output_dir']} ...")
        best_model_path = os.path.join(output_dir, "best_model.pth")

        torch.save({
            "params": self.params
            , "model_state": self.model.state_dict()
        }, best_model_path)
        save_code(os.getcwd(), output_dir)

        return best_model_path
    
    def train(self):
        best_result, best_model_path = float("inf"), None
        for epoch in range(self.params["pretrain_epoch"]):
            self.model.train()
            pretrain_loss = []
            for inputs in (pbar := tqdm(self.pretrain_loader, ncols=100)):
                pbar.set_description(f"{'Pre-Train':<{10 if self.params['do_pretrain'] else 5}} [{epoch + 1:2d}/{self.params['pretrain_epoch']:2d}] (Loss={np.mean(pretrain_loss) if pretrain_loss else 0.0:.3f})")
                inputs = move_to_cuda(inputs)
                self.optimizer.zero_grad()
                with amp.autocast():
                    loss = self.model(inputs, pretrain=True)
                    pretrain_loss.append(loss.item())
                if torch.isnan(loss):
                    raise ValueError("Loss of training is NAN.")
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            pretrain_loss = np.mean(pretrain_loss) if pretrain_loss else 0.0

            logger.info(f"Pre-Training of Epoch [{epoch + 1:3d}/{self.params['pretrain_epoch']:3d}] Pre-Training Loss={pretrain_loss:.3f}")
            
            if pretrain_loss < best_result:
                best_result = pretrain_loss
                best_model_path = self.save_best_model()
        
        return best_model_path


class Trainer(object):
    def __init__(self, params, model, train_dataset, dev_dataset):
        self.params = params
        logger.info("Inititalizing Trainer ...")

        self.train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=self.params["train_shuffle"])
        self.dev_loader = DataLoader(dev_dataset, batch_size=self.params["batch_size"])

        self.model = model.cuda()
        self.optimizer = getattr(torch.optim, self.params["optim"])(
            filter(lambda p: p.requires_grad, self.model.parameters())
            , lr=self.params["lr"]
            , weight_decay=self.params["weight_decay"]
        )
        self.scaler = amp.GradScaler()

        model_size, model_num = calc_model_size(self.model)
        logger.info(f"[-] Number of Model Parameters={model_num} & Model Size={model_size}.")
    
    def save_best_model(self):
        output_dir = os.path.join(self.params["output_dir"], "train")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        logger.info(f"  [|] Saving Best Model to {self.params['output_dir']} ...")
        best_model_path = os.path.join(output_dir, "best_model.pth")

        torch.save({
            "params": self.params
            , "model_state": self.model.state_dict()
        }, best_model_path)
        save_code(os.getcwd(), output_dir)

        return best_model_path

    def save_status(self, vids, predictions, references, representations, attentions):
        predictinos_path = os.path.join(self.params["output_dir"], "predictions.pkl")
        pickle.dump({
            "vids": vids
            , "predictions": predictions
            , "references": references
        }, open(predictinos_path, "wb"))

        representations_path = os.path.join(self.params["output_dir"], "representations.pkl")
        pickle.dump(representations, open(representations_path, "wb"))

        attentions_path = os.path.join(self.params["output_dir"], "attentions.pkl")
        pickle.dump(attentions, open(attentions_path, "wb"))

    def train(self):
        best_result, best_model_path = float("-inf"), None
        for epoch in range(self.params["epoch"]):
            self.model.train()
            train_loss = []
            for inputs, targets in (pbar := tqdm(self.train_loader, ncols=100)):
                pbar.set_description(f"{'Train':<{10 if self.params['do_pretrain'] else 5}} [{epoch + 1:2d}/{self.params['epoch']:2d}] (Loss={np.mean(train_loss) if train_loss else 0:.3f})")
                inputs, targets = move_to_cuda(inputs), move_to_cuda(targets)
                self.optimizer.zero_grad()
                with amp.autocast():
                    (loss, logits), (mrepr, attn) = self.model(inputs, targets)
                    train_loss.append(loss.item())
                if torch.isnan(loss):
                    raise ValueError("Loss of training is NAN.")
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            train_loss = np.mean(train_loss)
            
            self.model.eval()
            eval_loss, predictions, references = [], [], []
            for inputs, targets in (pbar := tqdm(self.dev_loader, ncols=100)):
                pbar.set_description(f"{'Dev':<{10 if self.params['do_pretrain'] else 5}} [{epoch + 1:2d}/{self.params['epoch']:2d}] (Loss={np.mean(eval_loss) if eval_loss else 0:.3f})")
                inputs, targets = move_to_cuda(inputs), move_to_cuda(targets)
                with torch.no_grad():
                    with amp.autocast():
                        (loss, logits), (mrepr, attn) = self.model(inputs, targets)
                        eval_loss.append(loss.item())
                predictions.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
                references.extend(targets.detach().cpu().tolist())
            eval_loss = np.mean(eval_loss)
            
            dev_metrics = calc_metrics(predictions, references)
            if "-" in self.params["dev_metric"]:
                _avg, _mtc = self.params["dev_metric"].split("-")
                dev_result = dev_metrics[_avg][_mtc]
            elif self.params["dev_metric"] == "acc":
                dev_result = dev_metrics["acc"]
            else:
                raise NotImplementedError
            logger.info(f"Training of Epoch [{epoch + 1:3d}/{self.params['epoch']:3d}] Train Loss={train_loss:.3f} Dev Metrics: Loss={eval_loss:.3f} {inline_format_metrics(dev_metrics)}")

            if dev_result > best_result:
                best_result = dev_result
                best_model_path = self.save_best_model()
        
        return best_model_path

    def evaluate(self, eval_dataset, save_status=False):
        eval_loader = DataLoader(eval_dataset, batch_size=self.params["batch_size"])

        self.model.eval()
        predictions, references, representations, attentions = [], [], [], []
        for inputs, targets in tqdm(eval_loader, desc="Eval", ncols=100):
            inputs, targets = move_to_cuda(inputs), move_to_cuda(targets)
            with torch.no_grad():
                with amp.autocast():
                    (loss, logits), (mrepr, attn) = self.model(inputs, targets)
            representations.append(mrepr.detach().cpu().numpy())
            attentions.append(attn.detach().cpu().numpy())
            predictions.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            references.extend(targets.detach().cpu().tolist())
        representations = np.concatenate(representations, axis=0)
        attentions = np.stack(attentions, axis=0).mean(axis=0)
        
        if save_status:
            self.save_status(eval_loader.dataset.vids, predictions, references, representations, attentions)
        eval_result = calc_metrics(predictions, references)
        return eval_result


def main():
    params = load_default_params()
    params = overriding_params(params=params)
    params = complete_params(params=params)

    set_seed(params["seed"])
    torch.cuda.set_device(params["cuda"])
    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"])
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s]: %(message)s"
        , datefmt="%Y.%m.%d-%H:%M:%S"
        , level=logging.INFO
        , filename=os.path.join(params["output_dir"], f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
        , filemode="w"
    )
    show_params(params)

    if params["do_pretrain"]:
        result_path = os.path.normpath(params["output_dir"]).split(os.sep)
        for i in range(len(result_path)):
            if result_path[i].startswith("modal"):
                result_path[i] = "*"
            elif "=" in result_path[i]:
                if result_path[i].startswith("epoch"):
                    result_path[i] = "*"
        result_path[-1] = "*"
        result_path = os.path.join(os.sep.join(result_path), "pretrain", "best_model.pth")
        pretrain_best_model_path_list = list(glob(result_path))
        if len(pretrain_best_model_path_list) == 0:
            logger.info("Loading Pre-Training Dataset ...")
            pretrain_dataset = Dataset(params, "pretrain")
            logger.info(f"[+] Pre-Train Size: {len(pretrain_dataset)}")
            logger.info("Loading Model ...")
            model = CVLA(params).cuda()
            trainer = PreTrainer(params, model, pretrain_dataset)
            logging.info("Pre-Training ...")
            pretrain_best_model_path = trainer.train()
        else:
            logger.info("Skip Pre-Training ...")
            pretrain_best_model_path = pretrain_best_model_path_list[0]
            logger.info(f"Pre-Trained Model Path: {pretrain_best_model_path}")

    logger.info("Loading Dataset ...")
    train_dataset = Dataset(params, "train")
    dev_dataset = Dataset(params, "dev")
    test_dataset = Dataset(params, "test")
    logger.info(f"[+] Train Size: {len(train_dataset)} Dev Size: {len(dev_dataset)} Test Size: {len(test_dataset)}")
    logger.info("Loading Model ...")
    model = CVLA(params).cuda()
    trainer = Trainer(params, model, train_dataset, dev_dataset)
    if params["do_pretrain"]:
        trainer.model.load_state_dict(torch.load(pretrain_best_model_path, map_location=torch.device(f"cuda:{params['cuda']}"))["model_state"])
    if params["do_train"]:
        logging.info("Training ...")
        train_best_model_path = trainer.train()
    else:
        train_best_model_path = list(glob(os.path.join(params["output_dir"], os.pardir, "*", "train", "best_model.pth")))
        assert len(train_best_model_path) != 0
        train_best_model_path = train_best_model_path[0]
    trainer.model.load_state_dict(torch.load(train_best_model_path, map_location=torch.device(f"cuda:{params['cuda']}"))["model_state"])

    logging.info("Evaluating ...")
    dev_result = trainer.evaluate(dev_dataset)
    logging.info(f"[*] Best Dev Result: {inline_format_metrics(dev_result)}")
    logging.info("Testing ...")
    test_result = trainer.evaluate(test_dataset, save_status=True)
    logging.info(f"[#] Test Result: {inline_format_metrics(test_result)}")


if __name__ == "__main__":
    ignore_warnings()
    main()
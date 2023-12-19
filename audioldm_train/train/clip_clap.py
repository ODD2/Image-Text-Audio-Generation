# Author: Haohe Liu
# Email: haoheliu@gmail.com
# Date: 11 Feb 2023

import sys

sys.path.append("src")
import shutil
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import yaml
import torch
import wandb
import numpy as np

from tqdm import tqdm
from pytorch_lightning.strategies.ddp import DDPStrategy
from audioldm_train.utilities.data.dataset import MusicDataset

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from audioldm_train.utilities.tools import (
    get_restore_step,
    copy_test_subset_data,
)
from audioldm_train.utilities.model_util import instantiate_from_config
import logging
from sklearn.metrics import top_k_accuracy_score
logging.basicConfig(level=logging.WARNING)


def print_on_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main(configs, config_yaml_path, exp_group_name, exp_name, perform_validation, debug=False):
    if "seed" in configs.keys():
        seed_everything(configs["seed"])
    else:
        print("SEED EVERYTHING TO 0")
        seed_everything(0)

    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(
            configs["precision"]
        )  # highest, high, medium

    log_path = configs["log_directory"]
    batch_size = configs["model"]["params"]["batchsize"]

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    dataset = MusicDataset(configs, split="train", add_ons=dataloader_add_ons)
    batch_size = 256
    loader = DataLoader(
        dataset,
        batch_size=8 if debug else batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )

    print(
        "The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s"
        % (len(dataset), len(loader), batch_size)
    )

    val_dataset = MusicDataset(configs, split="val", add_ons=dataloader_add_ons)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=True
    )

    clip_clap = instantiate_from_config(configs["model"])
    clip_clap.unconditional_prob = 0.0
    clip_clap = clip_clap.to("cuda")

    optimizer = torch.optim.AdamW(clip_clap.parameters(), lr=3e-4)

    # init wandb
    wandb.init(
        project="audioldm_clipclap",
        mode=("offline"if debug else "online")
    )
    wandb.watch(models=clip_clap, log="gradients", log_freq=100)

    global_step = 0
    best_performance = 0
    for epoch in range(100):
        # train
        clip_clap.clip.train()
        pbar = tqdm(loader)
        wandb.log({"lr": get_lr(optimizer), "epoch": epoch}, step=global_step)
        for data in pbar:
            results = clip_clap.three_modal_contrastive_loss(dict(
                image=data["image"].to(memory_format=torch.contiguous_format, device="cuda"),
                audio=data["waveform"].to(memory_format=torch.contiguous_format, device="cuda").float(),
                text=list(data["text"])
            ))
            loss = results["ce_loss"] + results["mse_loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"training loss: {loss}")
            # metrics
            labels = results["labels"]
            i2a_probs = results["i2a_probs"]
            i2t_probs = results["i2t_probs"]
            a2t_probs = results["a2t_probs"]

            i2a_top3 = top_k_accuracy_score(labels, i2a_probs, k=3)
            i2t_top3 = top_k_accuracy_score(labels, i2t_probs, k=3)
            a2t_top3 = top_k_accuracy_score(labels, a2t_probs, k=3)

            metrics = dict(
                loss=loss.detach(),
                ce_loss=results["ce_loss"].detach(),
                mse_loss=results["mse_loss"].detach(),
                i2a_top3=i2a_top3,
                i2t_top3=i2t_top3,
                a2t_top3=a2t_top3
            )
            wandb.log(
                {
                    f"train/{k}": v
                    for k, v in metrics.items()
                },
                step=global_step
            )

            global_step += 1

        # validation
        clip_clap.clip.eval()
        with torch.no_grad():
            total_loss = 0
            total_samples = 0
            i2a_probs = []
            i2t_probs = []
            a2t_probs = []
            labels = []
            pbar = tqdm(val_loader)
            for data in pbar:
                results = clip_clap.three_modal_contrastive_loss(dict(
                    image=data["image"].to(memory_format=torch.contiguous_format, device="cuda"),
                    audio=data["waveform"].to(memory_format=torch.contiguous_format, device="cuda").float(),
                    text=list(data["text"])
                ))
                loss = results["ce_loss"] + results["mse_loss"]
                pbar.set_description(f"training loss: {loss}")

                # metrics
                total_samples += data["image"].shape[0]
                total_loss += loss.item() * data["image"].shape[0]
                labels.append(results["labels"])
                i2a_probs.append(results["i2a_probs"])
                i2t_probs.append(results["i2t_probs"])
                a2t_probs.append(results["a2t_probs"])
            i2a_probs = np.concatenate(i2a_probs, axis=0)
            i2t_probs = np.concatenate(i2t_probs, axis=0)
            a2t_probs = np.concatenate(a2t_probs, axis=0)
            labels = np.concatenate(labels, axis=0)
            i2a_top = top_k_accuracy_score(labels, i2a_probs, k=3)
            i2t_top = top_k_accuracy_score(labels, i2t_probs, k=3)
            a2t_top = top_k_accuracy_score(labels, a2t_probs, k=3)
            metrics = dict(
                loss=total_loss / total_samples,
                i2a_top=i2a_top,
                i2t_top=i2t_top,
                a2t_top=a2t_top
            )
            wandb.log(
                {
                    f"valid/{k}": v
                    for k, v in metrics.items()
                },
                step=global_step
            )

            performance = i2t_top + i2a_top
            if (performance > best_performance):
                ckpt_folder = os.path.join("log/checkpoints", wandb.run.id)
                best_performance = performance
                os.makedirs(ckpt_folder, exist_ok=True)
                torch.save(
                    {
                        k: v
                        for k, v in clip_clap.named_parameters()
                        if v.requires_grad
                    },
                    os.path.join(ckpt_folder, "best_model.pt")
                )
                best_performance = performance

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        required=False,
        help="path to config .yaml file",
    )

    parser.add_argument(
        "--reload_from_ckpt",
        type=str,
        required=False,
        default=None,
        help="path to pretrained checkpoint",
    )

    parser.add_argument(
        "--debug",
        action="store_true"
    )

    parser.add_argument("--val", action="store_true")

    args = parser.parse_args()

    perform_validation = args.val

    assert torch.cuda.is_available(), "CUDA is not available"

    config_yaml = args.config_yaml

    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml_path = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)

    if args.reload_from_ckpt is not None:
        config_yaml["reload_from_ckpt"] = args.reload_from_ckpt

    if perform_validation:
        config_yaml["model"]["params"]["cond_stage_config"][
            "crossattn_audiomae_generated"
        ]["params"]["use_gt_mae_output"] = False
        config_yaml["step"]["limit_val_batches"] = None

    main(config_yaml, config_yaml_path, exp_group_name, exp_name, perform_validation, debug=args.debug)

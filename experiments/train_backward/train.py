# Copyright (c) 2024, Hitachi, Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import glob
import json
import os
import random
from collections import OrderedDict
from logging import INFO, StreamHandler, getLogger
from typing import List

import datasets
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel, get_scheduler

torch.multiprocessing.set_sharing_strategy("file_system")


def train(
    model,
    loader: DataLoader,
    optimizer,
    epoch: int,
    accelerator: Accelerator,
    lr_scheduler,
) -> float:
    model.train()
    log = {"loss": 0}
    with tqdm(
        enumerate(loader), total=len(loader), disable=not accelerator.is_main_process
    ) as pbar:
        for _, batch in pbar:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                log["loss"] += loss.item()
                if accelerator.is_main_process:
                    pbar.set_description(f"[Epoch {epoch}] [TRAIN]")
                    pbar.set_postfix(
                        OrderedDict(
                            loss=loss.item(),
                            lr=optimizer.optimizer.param_groups[0]["lr"],
                        )
                    )
    return {k: v / len(loader) for k, v in log.items()}


def valid(model, loader: DataLoader, epoch: int, accelerator: Accelerator) -> float:
    model.eval()
    log = {"loss": 0}
    with torch.no_grad():
        with tqdm(
            enumerate(loader),
            total=len(loader),
            disable=not accelerator.is_main_process,
        ) as pbar:
            for _, batch in pbar:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    log["loss"] += loss.item()
                    if accelerator.is_main_process:
                        pbar.set_description(f"[Epoch {epoch}] [VALID]")
                        pbar.set_postfix(OrderedDict(loss=loss.item()))
    return {k: v / len(loader) for k, v in log.items()}


def load_multi_arrow_files(files: List[str]):
    print(files)
    dataset = datasets.concatenate_datasets([Dataset.from_file(f) for f in files])
    return dataset


def main(args):
    config = (
        json.load(open(os.path.join(args.restore_dir, "training_state.json")))
        if args.restore_dir
        else {"argparse": dict()}
    )
    current_epoch = config.get("current_epoch", -1) + 1
    min_valid_loss = config.get("min_valid_loss", float("inf"))
    seed = config["argparse"].get("seed", args.seed)
    max_len = config["argparse"].get("max_len", args.max_len)
    log_dict = (
        json.load(open(os.path.join(args.restore_dir, "../log.json")))
        if args.restore_dir
        else dict()
    )
    logger = getLogger(__name__)
    handler = StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(INFO)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if args.restore_dir is not None:
        model = GPT2LMHeadModel.from_pretrained(args.restore_dir)
        tokenizer = AutoTokenizer(args.restore_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
        config = AutoConfig.from_pretrained(
            args.arch_id,
            vocab_size=len(tokenizer),
            n_ctx=max_len,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = GPT2LMHeadModel(config)
        model_size = sum(t.numel() for t in model.parameters())
        logger.info(f"Model size: {model_size/1000**2:.1f}M parameters")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    train_dataset = load_multi_arrow_files(
        glob.glob(os.path.join(args.train_dataset_dir, "data-*.arrow"))
    )
    valid_dataset = load_multi_arrow_files(
        glob.glob(os.path.join(args.valid_dataset_dir, "data-*.arrow"))
    )
    logger.info("Datasets are loaded")
    logger.info(f"Train dataset: {len(train_dataset)}")
    logger.info(f"Valid dataset: {len(valid_dataset)}")

    def collate_fn(batch):
        d = {}
        labels = [e["labels"] for e in batch]
        d["labels"] = torch.tensor(labels)
        input_ids = [e["input_ids"] for e in batch]
        d["input_ids"] = torch.tensor(input_ids)
        attention_mask = [e["attention_mask"] for e in batch]
        d["attention_mask"] = torch.tensor(attention_mask)
        return d

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.accumulation,
        num_training_steps=len(train_loader) * args.epochs,
    )
    best_path = os.path.join(args.outdir, "best")
    last_path = os.path.join(args.outdir, "last")
    os.makedirs(best_path, exist_ok=True)
    os.makedirs(last_path, exist_ok=True)
    tokenizer.save_pretrained(best_path)
    tokenizer.save_pretrained(last_path)
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation)
    model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, lr_scheduler
    )
    logger.info("Start of training...")
    accelerator.wait_for_everyone()
    for epoch in range(current_epoch, args.epochs):
        train_log = train(
            model, train_loader, optimizer, epoch, accelerator, lr_scheduler
        )
        valid_log = valid(model, valid_loader, epoch, accelerator)
        log_dict[f"Epoch {epoch}"] = {"train_log": train_log, "valid_log": valid_log}
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if min_valid_loss > valid_log["loss"]:
                # Save the best chckpoint
                accelerator.unwrap_model(model).save_pretrained(best_path)
                min_valid_loss = valid_log["loss"]
                training_state = {
                    "current_epoch": epoch,
                    "min_valid_loss": min_valid_loss,
                    "argparse": args.__dict__,
                }
                with open(os.path.join(best_path, "training_state.json"), "w") as fp:
                    json.dump(training_state, fp, indent=4)
            # Save checkpoint as the last checkpoint for each epoch
            accelerator.unwrap_model(model).save_pretrained(last_path)
            training_state = {
                "current_epoch": epoch,
                "min_valid_loss": min_valid_loss,
                "argparse": args.__dict__,
            }
            with open(os.path.join(last_path, "training_state.json"), "w") as fp:
                json.dump(training_state, fp, indent=4)
            with open(os.path.join(args.outdir, "log.json"), "w") as fp:
                json.dump(log_dict, fp, indent=4)
    print("Finish")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_dir", required=True)
    parser.add_argument("--valid_dataset_dir", required=True)
    parser.add_argument("--arch_id", default="gpt2")
    parser.add_argument("--tokenizer_id", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--outdir", default="models/sample/")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--accumulation", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--restore_dir", default=None)
    parser.add_argument("--num_warmup_steps", type=int, default=2000)
    parser.add_argument(
        "--lr_scheduler_type",
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)

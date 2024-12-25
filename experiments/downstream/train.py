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
import datetime
import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from conlleval import evaluate
from dataset import generate_dataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, SchedulerType, get_scheduler
from utils import load_label2id
from visualize.visualize_prediction import visualize

from backward_llm import (
    CausalLMForTokenClassification,
    CausalLMForTokenClassificationWithTrans,
    ConcatLMConfig,
    ConcatLMForTokenClassification,
    ReversedTokenizer,
)


def train(
    model,
    loader: DataLoader,
    optimizer,
    epoch: int,
    accelerator: Accelerator,
    lr_scheduler,
) -> float:
    model.train_mode()
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


def test(loader, model=None, epoch=0):
    pred_ids = []
    gold_ids = []
    loss = 0
    model.eval()
    with torch.no_grad():
        with tqdm(enumerate(loader), total=len(loader)) as pbar:
            for _, batch in pbar:
                batch = {k: v.cuda() for k, v in batch.items()}
                output = model(**batch)
                loss += output.loss.item()
                idx = batch["labels"] != -100
                pids = torch.argmax(output.logits, dim=-1)[idx].view(-1).cpu().tolist()
                gids = batch["labels"][idx].view(-1).cpu().tolist()
                pred_ids += pids
                gold_ids += gids
                pbar.set_description(f"[Epoch {epoch}] [VALID/TEST]")
                pbar.set_postfix(OrderedDict(loss=output.loss.item()))
    loss = loss / len(loader)
    pred_labels = [model.config.id2label[l] for l in pred_ids]
    gold_labels = [model.config.id2label[l] for l in gold_ids]
    pred_ids = torch.tensor(pred_ids)
    gold_ids = torch.tensor(gold_ids)
    cls_report = classification_report(
        y_true=gold_labels, y_pred=pred_labels, output_dict=True
    )
    try:
        p, r, f1 = evaluate(gold_labels, pred_labels)
    except ValueError:
        p, r, f1 = -1, -1, -1
    return {"loss": loss, "cls_report": cls_report, "span-f1": f1}


def main(args):
    config = (
        json.load(open(os.path.join(args.restore_dir, "training_state.json")))
        if args.restore_dir
        else {"argparse": dict()}
    )
    current_epoch = config.get("current_epoch", -1) + 1
    max_f1 = config.get("max_f1", 0)
    seed = config["argparse"].get("seed", args.seed)
    max_len = config["argparse"].get("max_len", args.max_len)
    log_dict = (
        json.load(open(os.path.join(args.restore_dir, "../log.json")))
        if args.restore_dir
        else dict()
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if args.restore_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.name_or_path_forward, add_prefix_space=True, local_files_only=True
        )
        r_tokenizer = ReversedTokenizer(
            args.name_or_path_forward, add_prefix_space=True, local_files_only=True
        )
        if "bert" in args.name_or_path_forward:
            tokenizer.pad_token = "[PAD]"
        else:
            tokenizer.pad_token = tokenizer.eos_token

    # Load dataset and get the mapping of id and tags
    dataset_args = {
        "dataset_name": args.dataset_name,
        "tokenizer": tokenizer,
        "r_tokenizer": r_tokenizer,
        "max_len": max_len,
        "split": "train",
        "few_shot": args.few_shot,  # If -1, all of data will be used.
        "task_name": args.task_name,
    }
    train_dataset = generate_dataset(**dataset_args)
    id2label = train_dataset["id2label"]
    train_dataset = train_dataset["dataset"]
    dataset_args["split"] = "validation"
    dataset_args["few_shot"] = -1
    valid_dataset = generate_dataset(**dataset_args)["dataset"]
    dataset_args["split"] = "test"
    test_dataset = generate_dataset(**dataset_args)["dataset"]

    model_class = {
        "concat": ConcatLMForTokenClassification,
        "forward": CausalLMForTokenClassification,
        "forward-trans": CausalLMForTokenClassificationWithTrans,
    }[args.representation]

    if args.restore_dir is not None:
        # If restore_dir is specified, we load the model from the path.
        model = model_class.from_pretrained(args.restore_dir)
    else:
        # label2id = load_label2id(args.dataset_name)
        # If restore_dir is None, we randomly initizalize the model.
        config = ConcatLMConfig(
            name_or_path_forward=args.name_or_path_forward,
            name_or_path_backward=args.name_or_path_backward,
            label2id={v: k for k, v in id2label.items()},
            id2label=id2label,
            max_length=args.max_len,
            dropout=args.dropout,
        )
        model = model_class(config)
        # The gradients of LM are not needed.
        for params in model.transformer.parameters():
            params.requires_grad = False
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
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
    accelerator.wait_for_everyone()
    for epoch in range(current_epoch, args.epochs):
        train_log = train(
            model, train_loader, optimizer, epoch, accelerator, lr_scheduler
        )
        valid_log = test(valid_loader, model, epoch)
        log_dict[f"Epoch {epoch}"] = {"train_log": train_log, "valid_log": valid_log}
        accelerator.wait_for_everyone()
        valid_score = valid_log["cls_report"]["accuracy"]
        if accelerator.is_main_process:
            if max_f1 < valid_score:
                # Save the best chckpoint
                accelerator.unwrap_model(model).save_pretrained(best_path)
                max_f1 = valid_score
                training_state = {
                    "current_epoch": epoch,
                    "max_f1": max_f1,
                    "argparse": args.__dict__,
                }
                with open(os.path.join(best_path, "training_state.json"), "w") as fp:
                    json.dump(training_state, fp, indent=4)
                test_log = test(test_loader, model, epoch)
                log_dict[f"Epoch {epoch}"]["test_log"] = test_log
            dt_now = datetime.datetime.now()
            log_dict[f"Epoch {epoch}"]["end_time"] = dt_now.isoformat()
            # Save checkpoint as the last checkpoint for each epoch
            accelerator.unwrap_model(model).save_pretrained(last_path)
            training_state = {
                "current_epoch": epoch,
                "max_f1": max_f1,
                "argparse": args.__dict__,
            }
            with open(os.path.join(last_path, "training_state.json"), "w") as fp:
                json.dump(training_state, fp, indent=4)
            with open(os.path.join(args.outdir, "log.json"), "w") as fp:
                json.dump(log_dict, fp, indent=4)
    visualize(
        loader=valid_loader,
        outfile=os.path.join(last_path, "output.txt"),
        model=model,
        tokenizer=tokenizer,
    )
    model.head.load_state_dict(torch.load(os.path.join(best_path, "pytorch_head.bin")))
    visualize(
        loader=valid_loader,
        outfile=os.path.join(best_path, "output.txt"),
        model=model,
        tokenizer=tokenizer,
    )
    print("Finish")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_or_path_forward", default="openai-community/gpt2")
    parser.add_argument("--name_or_path_backward", default=None)
    parser.add_argument("--dataset_name", default="conll2003")
    parser.add_argument("--task_name", default="ner", choices=["ner", "chunk", "pos"])
    parser.add_argument("--outdir", default="models/sample/")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--restore_dir", default=None)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--few_shot", type=int, default=-1)
    parser.add_argument("--representation", default="concat")
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
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

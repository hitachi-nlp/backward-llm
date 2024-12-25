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
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, SchedulerType, get_scheduler

from backward_llm import (
    CausalLMForTokenClassification,
    ConcatLMConfig,
    ConcatLMForTokenClassification,
    ConcatLMForTokenClassificationWithTrans,
    ReversedTokenizer,
)


def visualize(
    loader, outfile="sample.txt", model=None, restore_dir=None, tokenizer=None
):
    if model is None:
        model = CausalLMForTokenClassification.from_pretrained(restore_dir)
        tokenizer = AutoTokenizer.from_pretrained(restore_dir)
    id2label = model.config.id2label
    tokens = []
    pred_labels = []
    gold_labels = []
    for batch in tqdm(loader):  # Batch loop
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        pred_ids = torch.argmax(outputs.logits, dim=-1)
        gold_ids = batch["labels"]
        # Sentence loop
        for pid, gid, tid in zip(pred_ids, gold_ids, batch["input_ids"]):
            plabels = [
                id2label[l] for i, l in enumerate(pid.tolist()) if gid[i] != -100
            ]
            glabels = [
                id2label[l] for i, l in enumerate(gid.tolist()) if gid[i] != -100
            ]
            sent = tokenizer.decode(
                tid, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            token = sent.split(" ")[1:]
            tokens.append(token)
            pred_labels.append(plabels)
            gold_labels.append(glabels)

    # token_strs = []
    # pred_strs = []
    # gold_strs = []
    with open(outfile, "w") as fp:
        for plabels, glabels, token in zip(pred_labels, gold_labels, tokens):
            token_str = "|"
            pred_str = "|"
            gold_str = "|"
            for pl, gl, t in zip(plabels, glabels, token):
                if pl == "O":
                    pl = ""
                if gl == "O":
                    gl = ""
                max_len = max(len(pl), len(gl), len(t))
                token_str += t + " " * (max_len - len(t)) + "|"
                pred_str += pl + " " * (max_len - len(pl)) + "|"
                gold_str += gl + " " * (max_len - len(gl)) + "|"
            fp.write(token_str + "\n")
            fp.write(pred_str + "\n")
            fp.write(gold_str + "\n")
            fp.write("\n")


def visualize_for_crf(
    loader, outfile="sample.txt", model=None, restore_dir=None, tokenizer=None
):
    if model is None:
        model = CausalLMForTokenClassification.from_pretrained(restore_dir)
        tokenizer = AutoTokenizer.from_pretrained(restore_dir)
    id2label = model.config.id2label
    tokens = []
    pred_labels = []
    gold_labels = []
    for batch in tqdm(loader):  # Batch loop
        batch = {k: v.cuda() for k, v in batch.items()}
        pred_ids = model.decode(**batch)
        gold_ids = batch["labels"]
        # Sentence loop
        for pid, gid, tid in zip(pred_ids, gold_ids, batch["input_ids"]):
            plabels = [
                id2label[l] for i, l in enumerate(pid.tolist()) if gid[i] != -100
            ]
            glabels = [
                id2label[l] for i, l in enumerate(gid.tolist()) if gid[i] != -100
            ]
            sent = tokenizer.decode(
                tid, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            token = sent.split(" ")[1:]
            tokens.append(token)
            pred_labels.append(plabels)
            gold_labels.append(glabels)

    # token_strs = []
    # pred_strs = []
    # gold_strs = []
    with open(outfile, "w") as fp:
        for plabels, glabels, token in zip(pred_labels, gold_labels, tokens):
            token_str = "|"
            pred_str = "|"
            gold_str = "|"
            for pl, gl, t in zip(plabels, glabels, token):
                if pl == "O":
                    pl = ""
                if gl == "O":
                    gl = ""
                max_len = max(len(pl), len(gl), len(t))
                token_str += t + " " * (max_len - len(t)) + "|"
                pred_str += pl + " " * (max_len - len(pl)) + "|"
                gold_str += gl + " " * (max_len - len(gl)) + "|"
            fp.write(token_str + "\n")
            fp.write(pred_str + "\n")
            fp.write(gold_str + "\n")
            fp.write("\n")

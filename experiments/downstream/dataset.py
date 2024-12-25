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
import math
import os
import random

# from torch.utils.data import Dataset
from typing import List, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from .rtokenizer import ReversedTokenizer


class DatasetForConcatLM:
    def __init__(
        self,
        tokens: List[List[str]],
        labels: List[List[int]],
        tokenizer: PreTrainedTokenizer,
        r_tokenizer: ReversedTokenizer,
        max_len: int,
    ) -> None:
        self.tokens = tokens
        self.labels = labels
        self.tokenizer = tokenizer
        self.r_tokenizer = r_tokenizer
        self.max_len = max_len

    def __getitem__(self, idx: int) -> dict:
        tokens = self.tokens[idx]
        label = self.labels[idx]
        encode = self.tokenizer(
            tokens,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            is_split_into_words=True,
        )
        r_encode = self.r_tokenizer(
            tokens,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            is_split_into_words=True,
        )
        return {
            "input_ids": encode["input_ids"].squeeze(),
            "attention_mask": encode["attention_mask"].squeeze(),
            "input_ids_backward": r_encode["input_ids"].squeeze(),
            "labels": torch.tensor(label).squeeze(),
        }

    def __len__(self):
        return len(self.tokens)


def convert_subword_labels(
    sources: list[list[str]],
    tags: list[list[str]],
    tokenizer: PreTrainedTokenizer,
    max_len: int = 128,
) -> list[list[str]]:
    """This function converts the word-level tags into subword-level tags."""
    sub_labels = []
    for tokens, labels in zip(sources, tags):
        encode = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        labels_id = []
        prev_id = None
        for i in encode.word_ids(0):
            if i is None:
                labels_id.append(-100)
            elif i != prev_id:
                labels_id.append(labels[i])
            else:
                labels_id.append(-100)
            prev_id = i
        assert len(labels_id) == max_len
        sub_labels.append(labels_id)
    return sub_labels


# def tokenize_and_align_labels(examples):
#     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

#     labels = []
#     for i, label in enumerate(examples[f"ner_tags"]):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
#         previous_word_idx = None
#         label_ids = []
#         for word_idx in word_ids:  # Set the special tokens to -100.
#             if word_idx is None:
#                 label_ids.append(-100)
#             elif word_idx != previous_word_idx:  # Only label the first token of a given word.
#                 label_ids.append(label[word_idx])
#             else:
#                 label_ids.append(-100)
#             previous_word_idx = word_idx
#         labels.append(label_ids)

#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs


def load_few_shot_indices(sources, tags, k=4, verbose=False, id2label=None):
    entity2freq = {
        "-".join(l.split("-")[1:]): 0 for l in id2label.values()
    }  # E.g. I-ORG -> ORG
    indices = []
    for i in range(len(tags)):
        label = [id2label[l] for l in tags[i]]
        label_set = set(label)
        # The meaning of len(label_set) == 3 is to find like ['O', 'B-XXX', 'I-XXX'],
        #   which means an example that includes only one entity.
        if len(label_set) == 3 and any([l.startswith("I-") for l in label_set]):
            entity = "-".join(list(label_set - set(["O"]))[0].split("-")[1:])
            if entity2freq[entity] >= k:
                continue
            indices.append(i)
            entity2freq[entity] += 1
        # TODO: even if an example contains multiple entity, e.g. PER and ORG,
        #   we can use it as a single entity example by masking other entity
    print("Label distribution of few-shot:", entity2freq)
    if verbose:
        print("Selected indices of few-shot", indices)
    return indices


def generate_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    r_tokenizer: ReversedTokenizer,
    max_len: int,
    split="train",
    few_shot: int = -1,
    task_name="ner",
) -> DatasetForConcatLM:
    """
    This function recieves input file path(s) and returns a Dataset instance.
    """
    if dataset_name in ["conll2003", "wnut_17"]:
        raw_data = load_dataset(dataset_name, split=split)
        sources, tags = raw_data["tokens"], raw_data[f"{task_name}_tags"]
        id2label = {
            i: t
            for i, t in enumerate(
                raw_data.features[f"{task_name}_tags"].feature._int2str
            )
        }
    elif dataset_name == "fewnerd":
        split = {"train": "train", "validation": "dev", "test": "test"}[split]
        sources, tags, label2id, id2label = load_conll_format(
            f"data/few_nerd/supervised/{split}_bio.txt"
        )
    if few_shot > 0:
        indices = load_few_shot_indices(sources, tags, k=few_shot, id2label=id2label)
        sources = [sources[i] for i in indices]
        tags = [tags[i] for i in indices]
        # raw_data = raw_data.select(
        #     i for i in range(len(raw_data)) if i in indices
        # )
    subword_labels = convert_subword_labels(sources, tags, tokenizer, max_len)
    dataset = DatasetForConcatLM(
        tokens=sources,
        labels=subword_labels,
        tokenizer=tokenizer,
        r_tokenizer=r_tokenizer,
        max_len=max_len,
    )
    return {"dataset": dataset, "id2label": id2label}


def generate_dataset_from_processed(
    dataset_name, tokenizer, r_tokenizer, max_len: int, split: str, few_shot: int = -1
):
    data = Dataset.load_from_disk(os.path.join(dataset_name, split))
    if few_shot > 0:
        indices = load_few_shot_indices(data, k=few_shot)
        data = data.select(i for i in range(len(data)) if i in indices)
    return data


def load_conll_format(input_file):
    sources = []
    tags = []
    src_temp = []
    tag_temp = []
    with open(input_file) as fp:
        for line in fp:
            line = line.strip()
            if line == "":
                sources.append(src_temp)
                tags.append(tag_temp)
                src_temp = []
                tag_temp = []
                continue
            word, tag = line.split("\t")
            src_temp.append(word)
            tag_temp.append(tag)
    assert len(sources) == len(tags)
    tag_set = set([t for tag in tags for t in tag])
    id2label = {i: t for i, t in enumerate(tag_set)}
    label2id = {v: k for k, v in id2label.items()}
    for i in range(len(tags)):
        tags[i] = [label2id[t] for t in tags[i]]

    return sources, tags, label2id, id2label


if __name__ == "__main__":
    raw_dataset = load_dataset("conll2003", split="train")
    id2label = raw_dataset.features["ner_tags"].feature._int2str
    for k in [16, 64, 256, 512, 1024]:
        load_few_shot_indices(raw_dataset, k=k)
    # indices = load_few_shot_indices(raw_dataset, k=32)
    # few_data = raw_dataset.select(
    #     indices
    # )
    # print(len(few_data))
    # for d in few_data:
    #     print(d)
    #     print([id2label[t] for t in d['ner_tags']])
    #     print()

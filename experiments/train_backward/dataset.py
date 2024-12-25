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
from typing import List, Tuple

import torch
from datasets import load_dataset
from rtokenizer import ReversedTokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizer


class DatasetForReversedLLM:
    def __init__(
        self, tokens: List[List[str]], tokenizer: PreTrainedTokenizer, max_len: int
    ):
        self.tokens = tokens
        self.tokenizer = tokenizer
        self.len = len(self.tokens)
        self.max_len = max_len

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        tokens = self.tokens[idx]
        # sents = [self.text[i] for i in idx]
        encode = self.tokenizer.convert_tokens_to_ids(tokens)
        encode = torch.tensor(encode)
        return {"input_ids": encode.squeeze(), "labels": encode.squeeze()}
        # return {
        #     'input_ids': encode['input_ids'],
        #     'attention_mask': encode['attention_mask'],
        #     'labels': encode['input_ids']
        # }


def reverse_and_concat(text: List[str], tokenizer, length=1024):
    contexts = []
    current_tokens = []
    for document in tqdm(text):
        tokens = tokenizer.tokenize(document)
        for token in reversed(tokens):
            current_tokens.append(token)
            if len(current_tokens) == length:
                contexts.append(current_tokens[:])
                current_tokens = []
    return contexts


def filter_sent(sent: str):
    if sent == "":
        return False
    if sent.startswith(" ="):
        return False
    return True


def generate_dataset(
    tokenizer: PreTrainedTokenizer, max_len: int = 1024, split: str = "train"
):
    wiki_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    documents = [d["text"].rstrip() for d in wiki_dataset if filter_sent(d["text"])]
    if split == "train":
        bookcorpus = load_dataset("bookcorpus", split="train")
        bookcorpus = [d["text"].rstrip() for d in bookcorpus if filter_sent(d["text"])]
        documents += bookcorpus
    arranged_text = reverse_and_concat(documents, tokenizer, length=max_len)

    # For dummy
    # all_text = ["This is a sample sentences."] * 100
    return DatasetForReversedLLM(
        tokens=arranged_text, tokenizer=tokenizer, max_len=max_len
    )

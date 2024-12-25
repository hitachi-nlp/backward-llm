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
import os

import torch
from conlleval import evaluate
from dataset import generate_dataset
from modeling import (
    CausalLMForTokenClassification,
    ConcatLMConfig,
    ConcatLMForTokenClassification,
)
from rtokenizer import ReversedTokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForTokenClassification


def output_conll_format(dataset, gold_labels, pred_labels):
    output = ""
    tokens_list = dataset.tokens
    i = 0
    assert len(gold_labels) == sum([len(t) for t in tokens_list])
    for tokens in tokens_list:
        for t in tokens:
            output += f"{t} dummy {gold_labels[i]} {pred_labels[i]}\n"
            i += 1
        output += "\n"
    assert i == len(gold_labels)
    return output


def main(args):
    # model_class = CausalLMForTokenClassification if 'baseline' in args.name_or_path else ConcatLMForTokenClassification
    model_class = BertForTokenClassification
    model = model_class.from_pretrained(args.name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    r_tokenizer = ReversedTokenizer.from_pretrained(args.name_or_path)
    dataset_args = {
        "dataset_name": args.dataset_name,
        "tokenizer": tokenizer,
        "r_tokenizer": r_tokenizer,
        "max_len": model.config.max_length,
        "split": "test",
    }
    test_dataset = generate_dataset(**dataset_args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model.eval().cuda()
    gold_labels = []
    pred_labels = []
    id2label = model.config.id2label

    for batch in test_loader:
        if model_class != ConcatLMForTokenClassification:
            del batch["input_ids_backward"]
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        g_labels = batch["labels"].view(-1).cpu()
        p_labels = torch.argmax(output.logits, dim=-1).view(-1).cpu()
        idx = g_labels != -100
        pred_labels += p_labels[idx].view(-1).tolist()
        gold_labels += g_labels[idx].view(-1).tolist()
    print(pred_labels[:30])
    print(gold_labels[:30])
    assert len(pred_labels) == len(gold_labels)
    pred_labels = [id2label[l] for l in pred_labels]
    gold_labels = [id2label[l] for l in gold_labels]
    evaluate(gold_labels, pred_labels)

    output = output_conll_format(test_dataset, gold_labels, pred_labels)
    outdir = os.path.join(args.name_or_path, "sample.out")
    with open(outdir, "w") as f:
        f.write(output)


def test():
    evaluate(["O", "O", "B-temp", "O"], ["O", "O", "B-temp", "O"])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_or_path", default="gpt2")
    parser.add_argument("--dataset_name", default="conll2003")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--restore_dir", default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    # test()
    main(args)

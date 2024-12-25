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
import torch
from transformers import AutoTokenizer


class ReversedTokenizer:
    def __init__(self, name_or_path, **kwards):
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwards)
        if "bert" in name_or_path:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(cls, name_or_path, **kwards):
        return cls(name_or_path, **kwards)

    def save_pretrained(self, save_dir):
        self.tokenizer.save_pretrained(save_dir)

    def __call__(self, text, **kwards):
        """Returns tokenization results that have the same keys
        as the original one. But "input_ids" are reversed.

        Specifically, if the origina tokenizer returns
        input_ids = [1, 3, 2, 5, 0, 0], # where 0 is pad_token
        ReversedTokenizer convert it to
        [5, 2, 3, 1, 0, 0].

        For this converting
        [0,0,5,2,3,1] [0,0,1,1,1,1]

        Step1. Flip input_ids and attention_mask
            e.g. [1,3,2,5,0,0] => [0,0,5,2,3,1]
            e.g. [1,1,1,1,0,0] => [0,0,1,1,1,1]
        Step 2. Extract ids where attention_mask is 0 and 1 respectively
            e.g. [0,0,5,2,3,1] => [0,0] and [5,2,3,1]
        Step 3. Concatenate them reversed order.
            e.g. [5,2,3,1]+[0,0] => [5,2,3,1,0,0]
        """
        encode = self.tokenizer(text, **kwards)
        # Step 1
        r_input_ids = encode["input_ids"].flip(dims=[1])
        r_attention_mask = encode["attention_mask"].flip(dims=[1])
        r_encode = {"input_ids": None, "attention_mask": None}  # final result

        for ids, rmask, mask in zip(
            r_input_ids, r_attention_mask, encode["attention_mask"]
        ):
            # Step 2,3
            final_input_ids = torch.concat([ids[rmask == 1], ids[rmask == 0]], dim=-1)
            # accumulate result
            if r_encode["input_ids"] is None:
                r_encode["input_ids"] = final_input_ids.unsqueeze(dim=0)
                r_encode["attention_mask"] = mask.unsqueeze(
                    dim=0
                )  # note: not use reversed mask
            else:
                r_encode["input_ids"] = torch.vstack(
                    (r_encode["input_ids"], final_input_ids.unsqueeze(dim=0))
                )
                r_encode["attention_mask"] = torch.vstack(
                    (
                        r_encode["attention_mask"],
                        mask.unsqueeze(dim=0),  # note: not use reversed mask
                    )
                )
        return r_encode

    def __len__(self):
        return len(self.tokenizer)

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
import pytest
import torch
from transformers import AutoTokenizer

from backward_llm.rtokenizer import ReversedTokenizer


@pytest.mark.parametrize(
    ("sents", "max_len"),
    [
        (["This is sample sentences."], 20),
        (["This is sample sentences."], 128),
        (["This is sample sentences.", "This is also sample sentence"], 128),
    ],
)
def test_enoceding(sents, max_len):
    name_or_path = "openai-community/gpt2"
    normal_tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    normal_tokenizer.pad_token = normal_tokenizer.eos_token
    reversed_tokenizer = ReversedTokenizer.from_pretrained(name_or_path)
    tok_args = {
        "text": sents,
        "max_length": max_len,
        "padding": "max_length",
        "return_tensors": "pt",
        "truncation": True,
    }
    normal_encode = normal_tokenizer(**tok_args)
    reversed_encode = reversed_tokenizer(**tok_args)
    for k in normal_encode.keys():  # i.e. 'input_ids' and 'attention_mask'
        assert normal_encode[k].shape == reversed_encode[k].shape
        assert normal_encode[k].dtype == reversed_encode[k].dtype
    pad_token_id = normal_tokenizer.pad_token_id
    assert torch.all(
        (normal_encode["input_ids"] != pad_token_id)
        == (reversed_encode["input_ids"] != pad_token_id)
    )

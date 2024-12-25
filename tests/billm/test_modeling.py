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
from transformers import AutoModel, AutoTokenizer

from backward_llm.configuration import ConcatLMConfig
from backward_llm.modeling import (
    BaselineLMForTokenClassificationFromConcatLM,
    CausalLMForTokenClassification,
    CausalLMForTokenClassificationWithCRF,
    CausalLMForTokenClassificationWithTrans,
    ConcatLMForTokenClassification,
    ConcatLMForTokenClassificationWithCRF,
    ConcatLMForTokenClassificationWithTrans,
    ConcatLMModel,
)
from backward_llm.rtokenizer import ReversedTokenizer


def test_flipping_with_mask():
    config = ConcatLMConfig(
        name_or_path_forward="openai-community/gpt2",
        name_or_path_backward="openai-community/gpt2",
        num_labels=5,
    )
    model = ConcatLMModel(config)

    # assume (seq_len, hidden_size) = (6, 2)
    ids = torch.tensor([[4, 4], [3, 3], [2, 2], [1, 1], [-1, -1], [-1, -1]])
    # The last two tokens are padding token
    mask = torch.tensor([1, 1, 1, 1, 0, 0])
    assert torch.equal(
        model.flip_backward_result(ids, mask),
        torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [-1, -1], [-1, -1]]),
    )


def check_params_are_identical(params1, params2):
    for p1, p2 in zip(params1, params2):
        # print(p1)
        # print(p2)
        # print()
        assert torch.equal(p1, p2)
    return


def test_loading():
    config = ConcatLMConfig(
        name_or_path_forward="openai-community/gpt2",
        name_or_path_backward="openai-community/gpt2",
        num_labels=5,
    )

    gpt2_model = AutoModel.from_pretrained("openai-community/gpt2")

    # The test of check_params_are_identical() function.
    check_params_are_identical(gpt2_model.parameters(), gpt2_model.parameters())

    # check_params_are_identical(gpt2_model.parameters(), gpt2_model.parameters())
    causal_classes = [
        CausalLMForTokenClassification,
        CausalLMForTokenClassificationWithTrans,
    ]
    concat_classes = [
        ConcatLMForTokenClassification,
        ConcatLMForTokenClassificationWithTrans,
    ]
    for model_class in causal_classes:
        model = model_class(config)
        check_params_are_identical(
            model.transformer.parameters(), gpt2_model.parameters()
        )

    for model_class in concat_classes:
        model = model_class(config)
        check_params_are_identical(
            model.transformer.forward_lm.parameters(), gpt2_model.parameters()
        )
        check_params_are_identical(
            model.transformer.backward_lm.parameters(), gpt2_model.parameters()
        )


def test_load_save():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    id2label = {i: f"{i}" for i in range(9)}
    config = ConcatLMConfig(
        name_or_path_forward="openai-community/gpt2",
        name_or_path_backward="openai-community/gpt2",
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    print(config)

    # check_params_are_identical(gpt2_model.parameters(), gpt2_model.parameters())
    causal_classes = [
        CausalLMForTokenClassification,
        CausalLMForTokenClassificationWithTrans,
    ]
    concat_classes = [
        ConcatLMForTokenClassification,
        ConcatLMForTokenClassificationWithTrans,
    ]
    sample_inputs = {
        "input_ids": tokenizer("This is a sample sentence.", return_tensors="pt")[
            "input_ids"
        ],
        "input_ids_backward": tokenizer(
            "This is a sample sentence.", return_tensors="pt"
        )["input_ids"],
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[0, 1, 2, 3, 4, 4]]),
    }
    sample_outdir = "models/sample_load_save"
    for model_class in causal_classes:
        model = model_class(config)
        optimizer = torch.optim.AdamW(model.parameters())
        for params in model.transformer.parameters():
            params.requires_grad = False
        # なんでもいいので学習して重みに変化が生じた場合を想定
        loss = model(**sample_inputs).loss
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        # saveしてloadする
        model.save_pretrained(sample_outdir)
        loaded_model = model_class.from_pretrained(sample_outdir)
        # パラメタが同じか確認
        check_params_are_identical(model.parameters(), loaded_model.parameters())

    for model_class in concat_classes:
        model = model_class(config)
        optimizer = torch.optim.AdamW(model.parameters())
        for params in model.transformer.parameters():
            params.requires_grad = False
        loss = model(**sample_inputs).loss
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        model.save_pretrained(sample_outdir)
        loaded_model = model_class.from_pretrained(sample_outdir)
        check_params_are_identical(model.parameters(), loaded_model.parameters())


def test_crf_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "openai-community/gpt2", local_files_only=True
    )
    id2label = {i: f"{i}" for i in range(9)}
    config = ConcatLMConfig(
        name_or_path_forward="openai-community/gpt2",
        name_or_path_backward="openai-community/gpt2",
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    # print(config)

    sample_inputs = {
        "input_ids": tokenizer("This is a sample sentence.", return_tensors="pt")[
            "input_ids"
        ],
        "input_ids_backward": tokenizer(
            "This is a sample sentence.", return_tensors="pt"
        )["input_ids"],
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[0, 1, 2, 3, 4, 4]]),
    }
    sample_outdir = "models/sample_load_save"
    for model_class in [CausalLMForTokenClassificationWithCRF]:
        model = model_class(config)
        optimizer = torch.optim.AdamW(model.parameters())
        for params in model.transformer.parameters():
            params.requires_grad = False
        # Make some changes
        loss = model(**sample_inputs).loss
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        model.save_pretrained(sample_outdir)
        loaded_model = model_class.from_pretrained(sample_outdir)
        # パラメタが同じか確認
        check_params_are_identical(model.parameters(), loaded_model.parameters())
    del sample_inputs["labels"]
    del sample_inputs["input_ids_backward"]
    pred_tags = model.decode(**sample_inputs)
    print(pred_tags.shape, pred_tags)
    for tags in pred_tags.tolist():
        for t in tags:
            print(config.id2label[t])

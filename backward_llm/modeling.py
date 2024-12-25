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
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithCrossAttentions,
    TokenClassifierOutput,
)
from transformers.models.bert import BertLayer

from .configuration import CausalLMTokenClassificationConfig, ConcatLMConfig
from .crf import CRF


@dataclass
class ConcatLMOutput(CausalLMOutputWithCrossAttentions):
    concat_logits: torch.Tensor = None
    forward_output: CausalLMOutputWithCrossAttentions = None
    backward_output: CausalLMOutputWithCrossAttentions = None


class ClassificationHeadTwoLayers(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels):
        super().__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.activate = nn.ReLU()

    def forward(self, hidden_state):
        logits = self.hidden_layer(hidden_state)
        logits = self.classifier(self.activate(logits))
        return logits

    def init_weights(self):
        self._init_weights(self.hidden_layer)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class ClassficationHeadTransformerBlock(nn.Module):
    def __init__(self, input_size, num_labels):
        super().__init__()
        # We do not use BERT's pre-trained weight but use its configuration (e.g. hidden_size)
        bert_config = AutoConfig.from_pretrained("bert-base-cased")
        self.bert_layer = BertLayer(bert_config)
        self.hidden_size = bert_config.hidden_size
        self.hidden_layer = nn.Linear(input_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, hidden_state):
        logits = self.hidden_layer(hidden_state)
        trans_layer_outputs = self.bert_layer(logits)
        logits = self.classifier(trans_layer_outputs[0])
        return logits

    def init_weights(self):
        self._init_weights(self.hidden_layer)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class ClassificationHeadCRF(nn.Module):
    # https://github.com/kolloldas/torchnlp/blob/master/torchnlp/modules/crf.py
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels)
        self.activate = nn.ReLU()

    def forward(self, hidden_state, labels=None):
        logits = self.classifier(hidden_state)
        logits = self.classifier2(self.activate(hidden_state))
        if labels is not None:
            return self.crf.loss(logits, labels)
        else:
            return self.crf.forward(logits)


class ConcatLMModel(PreTrainedModel):
    """A model to caclulate concatenated representation of forward and backward model."""

    config_class = ConcatLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.forward_lm = AutoModel.from_pretrained(
            config.name_or_path_forward,
        )
        self.backward_lm = AutoModel.from_pretrained(
            config.name_or_path_backward,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids_backward: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        forward_output = self.forward_lm(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        backward_output = self.backward_lm(
            input_ids_backward,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        f_logits = forward_output.last_hidden_state  # (batch, seq_len, hidden_size)
        b_logits = backward_output.last_hidden_state  # (batch, seq_len, hidden_size)
        flipped_b_logits = None
        for mask, logits in zip(attention_mask, b_logits):
            # Backward model's hidden_states are flipped.
            # Thus we flip hidden_states again.
            # unsequeeze() is to change the shape (seq_len, hidden_size) => (1, seq_len, hidden_size)
            flipped_logits = self.flip_backward_result(logits, mask).unsqueeze(dim=0)
            if flipped_b_logits is None:
                flipped_b_logits = flipped_logits
            else:
                flipped_b_logits = torch.vstack((flipped_b_logits, flipped_logits))
        assert b_logits.shape == flipped_b_logits.shape
        concatenated_logits = torch.cat(
            [f_logits, flipped_b_logits], dim=-1
        )  # (batch, seq_len, hidden_size_forward+hidden_size_backward)
        return ConcatLMOutput(
            concat_logits=concatenated_logits,
            forward_output=forward_output,
            backward_output=backward_output,
        )

    def get_hidden_size(self):
        # the dimension of the hidden_state will be sum of forward's and backward's
        foward_dim = self.forward_lm.config.hidden_size
        backward_dim = self.backward_lm.config.hidden_size
        return foward_dim + backward_dim

    def flip_backward_result(self, hidden_state, attention_mask):
        """
        Args:
            1D tensor: A padded tensor like: [h_5, h_4, h_3, h_2, h_1, h_p, h_p, h_p]
            (Note that h_i is i-th hidden state, h_p is padding's hidden states)
        Return:
            1D fliped tensor like: [h_1, h_2, h_3, h_4, h_5, h_p, h_p, h_p]
        """
        # print(tensor.shape) -> (seq_len, hidden_size)
        valid_hidden = hidden_state[attention_mask == 1]
        other_hidden = hidden_state[attention_mask == 0]
        # We can recontruct input hidden by concat [valid, other] but
        new_hidden = torch.cat([valid_hidden.flip(dims=[0]), other_hidden], dim=0)
        return new_hidden


class ConcatLMForTokenClassification(PreTrainedModel):
    """A token classification model based on concatenated representation."""

    config_class = ConcatLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = ConcatLMModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.dropout)
        self.head = ClassificationHeadTwoLayers(
            self.transformer.get_hidden_size(),
            self.transformer.get_hidden_size(),
            self.num_labels,
        )
        self.post_init()

    def init_weights(self):
        self.head.init_weights()

    def train_mode(self):
        # This will be used instead of Model.train().
        # thepurpose istoalign the interface for each model.
        self.transformer.eval()
        self.dropout.train()
        self.head.train()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids_backward: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        with torch.no_grad():
            llm_output = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                input_ids_backward=input_ids_backward,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
        logits = llm_output.concat_logits
        logits = self.dropout(logits)

        logits = self.head(logits)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=llm_output.concat_logits
        )

    def save_pretrained(self, dir):
        self.config.save_pretrained(dir)
        torch.save(self.head.state_dict(), os.path.join(dir, "pytorch_head.bin"))

    @classmethod
    def from_pretrained(self, dir):
        config = ConcatLMConfig.from_pretrained(dir)
        model = ConcatLMForTokenClassification(config)
        model.head.load_state_dict(torch.load(os.path.join(dir, "pytorch_head.bin")))
        return model


class ConcatLMForTokenClassificationWithTrans(PreTrainedModel):
    """A token classification model based on concatenated representation,
    but the Transformer block is used as a head.
    """

    config_class = ConcatLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = ConcatLMModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.dropout)
        self.head = ClassficationHeadTransformerBlock(
            self.transformer.get_hidden_size(), self.num_labels
        )
        self.post_init()

    def init_weights(self):
        self.head.init_weights()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def train_mode(self):
        self.transformer.eval()
        self.dropout.train()
        self.head.train()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids_backward: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        with torch.no_grad():
            llm_output = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                input_ids_backward=input_ids_backward,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
        logits = llm_output.concat_logits
        logits = self.dropout(logits)
        logits = self.head(logits)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=llm_output.concat_logits
        )

    def save_pretrained(self, dir):
        self.config.save_pretrained(dir)
        torch.save(self.head.state_dict(), os.path.join(dir, "pytorch_head.bin"))

    @classmethod
    def from_pretrained(self, dir):
        config = ConcatLMConfig.from_pretrained(dir)
        model = ConcatLMForTokenClassificationWithTrans(config)
        model.head.load_state_dict(torch.load(os.path.join(dir, "pytorch_head.bin")))
        return model


class CausalLMForTokenClassification(PreTrainedModel):
    """A token classification model based on only forward model."""

    config_class = ConcatLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = AutoModel.from_pretrained(
            config.name_or_path_forward,
        )
        self.num_labels = config.num_labels
        self.head = ClassificationHeadTwoLayers(
            self.transformer.config.hidden_size,
            self.transformer.config.hidden_size,
            self.num_labels,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.post_init()

    def init_weights(self):
        self.head.init_weights()

    def train_mode(self):
        self.transformer.eval()
        self.head.train()
        self.dropout.train()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        **kwards,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        with torch.no_grad():
            output = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        logits = output.last_hidden_state
        logits = self.dropout(logits)
        logits = self.head(logits)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=output.last_hidden_state
        )

    def save_pretrained(self, dir):
        self.config.save_pretrained(dir)
        torch.save(self.head.state_dict(), os.path.join(dir, "pytorch_head.bin"))

    @classmethod
    def from_pretrained(self, dir):
        config = ConcatLMConfig.from_pretrained(dir)
        model = CausalLMForTokenClassification(config)
        model.head.load_state_dict(torch.load(os.path.join(dir, "pytorch_head.bin")))
        return model


class CausalLMForTokenClassificationWithTrans(PreTrainedModel):
    """A token classification model based on only forward model,
    but the Transformer block is used as a head.
    """

    config_class = ConcatLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = AutoModel.from_pretrained(
            config.name_or_path_forward,
        )
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.dropout)
        self.head = ClassficationHeadTransformerBlock(
            self.transformer.config.hidden_size, self.num_labels
        )
        self.post_init()

    def init_weights(self):
        self.head.init_weights()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def train_mode(self):
        self.transformer.eval()
        self.head.train()
        self.dropout.train()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        **kwards,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        with torch.no_grad():
            output = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        logits = output.last_hidden_state
        logits = self.dropout(logits)
        logits = self.head(logits)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=output.last_hidden_state
        )

    def save_pretrained(self, dir):
        self.config.save_pretrained(dir)
        torch.save(self.head.state_dict(), os.path.join(dir, "pytorch_head.bin"))

    @classmethod
    def from_pretrained(self, dir):
        config = ConcatLMConfig.from_pretrained(dir)
        model = CausalLMForTokenClassificationWithTrans(config)
        model.head.load_state_dict(torch.load(os.path.join(dir, "pytorch_head.bin")))
        return model


class BaselineLMForTokenClassificationFromConcatLM(PreTrainedModel):
    """concatモデル実装を使って最小限の編集でベースラインモデルを実現"""

    config_class = ConcatLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.concat_lm = ConcatLMModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = self.concat_lm.forward_lm.config.hidden_size
        self.head = ClassificationHeadTwoLayers(
            self.hidden_size, self.hidden_size, self.num_labels
        )
        self.post_init()

    def init_weights(self):
        self.head.init_weights()

    def train_mode(self):
        # This will be used instead of Model.train().
        # thepurpose istoalign the interface for each model.
        self.concat_lm.eval()
        self.dropout.train()
        self.head.train()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids_backward: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        with torch.no_grad():
            llm_output = self.concat_lm(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                input_ids_backward=input_ids_backward,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
        # Only forward information is used.
        logits = llm_output.forward_output.last_hidden_state
        logits = self.dropout(logits)
        logits = self.head(logits)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=llm_output.forward_output.last_hidden_state,
        )

    def save_pretrained(self, dir):
        self.config.save_pretrained(dir)
        torch.save(self.head.state_dict(), os.path.join(dir, "pytorch_head.bin"))

    @classmethod
    def from_pretrained(self, dir):
        config = ConcatLMConfig.from_pretrained(dir)
        model = BaselineLMForTokenClassificationFromConcatLM(config)
        model.head.load_state_dict(torch.load(os.path.join(dir, "pytorch_head.bin")))
        return model


class ConcatLMForTokenClassificationWithCRF(PreTrainedModel):
    """A token classification model based on concatenated representation."""

    config_class = ConcatLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.concat_lm = ConcatLMModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.dropout)
        self.head = ClassificationHeadCRF(
            self.concat_lm.get_hidden_size(), self.num_labels
        )
        # self.post_init()

    def init_weights(self):
        self.head.init_weights()

    def train_mode(self):
        # This will be used instead of Model.train().
        # thepurpose istoalign the interface for each model.
        self.concat_lm.eval()
        self.dropout.train()
        self.head.train()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids_backward: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        with torch.no_grad():
            llm_output = self.concat_lm(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                input_ids_backward=input_ids_backward,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
        logits = llm_output.concat_logits
        logits = self.dropout(logits)
        loss = None
        loss = self.head(logits, labels)
        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=llm_output.concat_logits
        )

    def decode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids_backward: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwards,
    ):
        with torch.no_grad():
            llm_output = self.concat_lm(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                input_ids_backward=input_ids_backward,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
        logits = llm_output.concat_logits
        tags = self.head(logits)
        return tags

    def save_pretrained(self, dir):
        self.config.save_pretrained(dir)
        torch.save(self.head.state_dict(), os.path.join(dir, "pytorch_head.bin"))

    @classmethod
    def from_pretrained(self, dir):
        config = ConcatLMConfig.from_pretrained(dir)
        model = ConcatLMForTokenClassification(config)
        model.head.load_state_dict(torch.load(os.path.join(dir, "pytorch_head.bin")))
        return model


class CausalLMForTokenClassificationWithCRF(PreTrainedModel):
    """A token classification model based on only forward model."""

    config_class = ConcatLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = AutoModel.from_pretrained(
            config.name_or_path_forward,
        )
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(p=0.1)
        self.head = ClassificationHeadCRF(
            self.transformer.config.hidden_size, self.num_labels
        )
        # self.post_init()

    def init_weights(self):
        self.head.init_weights()

    def train_mode(self):
        self.transformer.eval()
        self.head.train()
        self.dropout.train()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        **kwards,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        with torch.no_grad():
            output = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        logits = output.last_hidden_state
        logits = self.dropout(logits)
        loss = None
        loss = self.head(logits, labels)
        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=output.last_hidden_state
        )

    def decode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids_backward: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwards,
    ):
        output = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = output.last_hidden_state
        tags = self.head(logits)
        return tags

    def save_pretrained(self, dir):
        self.config.save_pretrained(dir)
        torch.save(self.head.state_dict(), os.path.join(dir, "pytorch_head.bin"))

    @classmethod
    def from_pretrained(self, dir):
        config = ConcatLMConfig.from_pretrained(dir)
        model = CausalLMForTokenClassificationWithCRF(config)
        model.head.load_state_dict(torch.load(os.path.join(dir, "pytorch_head.bin")))
        return model

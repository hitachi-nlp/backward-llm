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
from transformers import AutoConfig, PretrainedConfig


class ConcatLMConfig(PretrainedConfig):
    def __init__(
        self,
        name_or_path_forward: str = "gpt2",
        name_or_path_backward: str = "gpt2",
        dropout: float = 0.1,
        initializer_range: float = 0.02,
        **kwards,
    ):
        super().__init__(**kwards)
        self.name_or_path_forward = name_or_path_forward
        self.name_or_path_backward = name_or_path_backward
        self.dropout = dropout
        self.initializer_range = initializer_range


class CausalLMTokenClassificationConfig(PretrainedConfig):
    def __init__(
        self,
        lm_model: str = "gpt2",
        dropout: float = 0.1,
        initializer_range: float = 0.2,
        **kwards,
    ):
        super().__init__(**kwards)
        self.lm_model = lm_model
        self.dropout = dropout
        self.initializer_range = initializer_range

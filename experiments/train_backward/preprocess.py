# Copyright (c) 2020 The HuggingFace Inc. team.
# Copyright (c) 2024, Hitachi, Ltd.
# This file has been adopted from https://github.com/huggingface/transformers/blob/5c67682b169576c4859700d551090ff79d450a9a/examples/pytorch/language-modeling/run_clm.py
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
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import List, Optional

import datasets
import evaluate
import psutil
import transformers
from datasets import concatenate_datasets, load_dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.33.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[List[str]] = field(
        default_factory=list,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    base_save_dir: Optional[str] = field(default="./data")
    raw_dir: Optional[List[str]] = field(default_factory=list)
    tokenized_dir: Optional[List[str]] = field(default_factory=list)
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )

    validation_size: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The rate of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`"
            )

        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`train_file` should be a csv, a json or a txt file."
                    )
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`validation_file` should be a csv, a json or a txt file."
                    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets_list = []
        for ds_name, ds_config_name in zip(
            data_args.dataset_name, data_args.dataset_config_name
        ):
            if ds_config_name == "None":
                ds_config_name = None
            raw_datasets = load_dataset(
                ds_name, name=ds_config_name, streaming=data_args.streaming
            )

            if "validation" not in raw_datasets.keys():
                raw_datasets = raw_datasets["train"].train_test_split(
                    test_size=data_args.validation_size
                )
                if not "validation" in raw_datasets.keys():
                    raw_datasets["validation"] = raw_datasets.pop("test")
            raw_datasets_list.append(raw_datasets)

        # Concatenate datasets and shuffle
        for split in ["train", "validation"]:
            raw_datasets_list[0][split] = datasets.concatenate_datasets(
                [d[split] for d in raw_datasets_list]
            )
            if split == "train":
                raw_datasets_list[0][split].shuffle()
        del raw_datasets_list[1:]
        assert len(raw_datasets_list) == 1
        # Getting % usage of virtual_memory ( 3rd field)
        logger.info(f"after being concatenated: {raw_datasets_list[0]}")
        raw_dir = data_args.raw_dir[0]
        if not os.path.isdir(f"{data_args.base_save_dir}/{raw_dir}"):
            raw_datasets_list[0].save_to_disk(f"{data_args.base_save_dir}/{raw_dir}")
        logger.debug(f"append raw {ds_name}/{ds_config_name} data")

    # else:
    #    data_files = {}
    #    if data_args.train_file is not None:
    #        data_files["train"] = data_args.train_file
    #        extension = data_args.train_file.split(".")[-1]
    #    if data_args.validation_file is not None:
    #        data_files["validation"] = data_args.validation_file
    #        extension = data_args.validation_file.split(".")[-1]
    #    if extension == "txt":
    #        extension = "text"
    #    raw_datasets = load_dataset(
    #        extension,
    #        data_files=data_files,
    #        cache_dir=model_args.cache_dir,
    #        token=model_args.token,
    #    )

    #    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    #    if "validation" not in raw_datasets.keys():
    #        raw_datasets["validation"] = load_dataset(
    #            extension,
    #            data_files=data_files,
    #            split=f"train[:{data_args.validation_split_percentage}%]",
    #            cache_dir=model_args.cache_dir,
    #            token=model_args.token,
    #        )
    #        raw_datasets["train"] = load_dataset(
    #            extension,
    #            data_files=data_files,
    #            split=f"train[{data_args.validation_split_percentage}%:]",
    #            cache_dir=model_args.cache_dir,
    #            token=model_args.token,
    #        )

    tokenizer_kwargs = {
        "cache_dir": data_args.cache_dir,
        "use_fast": data_args.use_fast_tokenizer,
    }
    if data_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            data_args.tokenizer_name, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    text_column_name = "text"

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            max_seq_length = 1024
    else:
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line
                for line in examples[text_column_name]
                if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets_list = []
        with training_args.main_process_first(desc="dataset map tokenization"):
            for raw_datasets in raw_datasets_list:
                if not data_args.streaming:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=[text_column_name],
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on dataset line_by_line",
                    )
                else:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        remove_columns=[text_column_name],
                        batched=True,
                    )
                tokenized_datasets_list.append(tokenized_datasets)
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            print("RAM memory % used:", psutil.virtual_memory()[2])
            return tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            )

        tokenized_datasets_list = []
        with training_args.main_process_first(desc="dataset map tokenization"):
            for raw_datasets in raw_datasets_list:
                column_names = list(raw_datasets["train"].features)
                if not data_args.streaming:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on every text in dataset",
                    )
                else:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        remove_columns=column_names,
                    )

                logger.debug(
                    f"after being tokenized: {tokenized_datasets['train'].features}"
                )
                tokenized_datasets_list.append(tokenized_datasets)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [
                    t[i : i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)
                ]
                for k, t in concatenated_examples.items()
            }
            # Reverse input_ids. We assume that there is no padding so we can simply reverse each ids.
            result["input_ids"] = [ids[::-1] for ids in result["input_ids"]]
            # for CausalLM
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with training_args.main_process_first(desc="grouping texts together"):
            for i, tokenized_datasets in enumerate(tokenized_datasets_list):
                if not data_args.streaming:
                    tokenized_datasets_list[i] = tokenized_datasets.map(
                        group_texts,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc=f"Grouping texts in chunks of {max_seq_length}",
                    )
                else:
                    tokenized_datasets_list[i] = tokenized_datasets.map(
                        group_texts,
                        batched=True,
                    )

                logger.debug(
                    f"after being grouped: {tokenized_datasets_list[i]['train'].features}"
                )

    for tokenized_datasets, tokenized_dir in zip(
        tokenized_datasets_list, data_args.tokenized_dir
    ):
        if not os.path.isdir(f"{data_args.base_save_dir}/{tokenized_dir}"):
            tokenized_datasets.save_to_disk(
                f"{data_args.base_save_dir}/{tokenized_dir}"
            )


if __name__ == "__main__":
    main()

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

from datasets import load_dataset


def main(args):
    raw_dataset = load_dataset(args.dataset_name, split=args.split)
    id2label = raw_dataset.features["ner_tags"].feature._int2str
    token_strs = []
    tag_strs = []
    for tokens, tag_ids in zip(raw_dataset["tokens"], raw_dataset["ner_tags"]):
        tag_ids = [id2label[t] for t in tag_ids]
        token_str = ""
        tag_str = ""
        prev_tag = None
        for to, ta in zip(tokens, tag_ids):
            # Check the
            if prev_tag != ta.split("-")[-1]:
                token_str += "|"
                tag_str += "|"
            else:
                token_str += " "
                tag_str += " "
            max_len = max(len(to), len(ta))
            token_str += to + " " * (max_len - len(to))
            prev_tag = ta.split("-")[-1]
            if ta == "O":
                ta = ""  # O tag will be invisible
            tag_str += ta + " " * (max_len - len(ta))
        token_strs.append(token_str)
        tag_strs.append(tag_str)

    with open(args.outfile, "w") as fp:
        idx = 1
        for to_str, ta_str in zip(token_strs, tag_strs):
            fp.write(f"Line {idx}\n")
            idx += 1
            fp.write(to_str + "|\n")
            fp.write(ta_str + "|\n\n")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="conll2003")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--outfile", default="temp.out")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    # test()
    main(args)

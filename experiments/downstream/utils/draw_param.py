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
import json
import random


def main(args):
    # d = {"batch_size": [4, 8, 16, 32],
    # "lr": [1e-3, 3e-3, 5e-3, 7e-3, 1e-4, 3e-4, 5e-4, 7e-4, 1e-5],
    # "seed": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}
    # with open("test.json", "w") as fp:
    #     json.dump(d, fp, indent=2)
    # return
    assert args.json is not None
    params = json.load(open(args.json))
    params = params[args.key]
    random.shuffle(params)
    print(params[0])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    parser.add_argument("--key", default="lr")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)

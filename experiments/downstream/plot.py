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
import math

import matplotlib.pyplot as plt


def fewshot():
    xlabels = [1, 2, 3, 3.5, 4, -1]
    xlabels = "4^1 4^2 4^3 4^(3.5) 4^4 Full".split(" ")
    x = [math.log(xx, 4) for xx in [4, 16, 64, 128, 256, 14041]]
    # CoNLL2003
    bert = [5.160, 43.06, 67.18, 72.75, 77.53, 91.18]
    bert_large = [6.62, 30.75, 53.47, 60.91, 67.1, 84.86]
    gpt2_baseline = [9.900, 35.03, 52.01, 56.37, 60.37, 77.14]
    gpt2_concat = [27.19, 47.11, 63.35, 68.35, 72.38, 88.58]
    gpt2_large_baseline = [15.35, 36.27, 50.67, 56.8, 59.54, 79.26]
    gpt2_large_concat = [27.92, 46.94, 64.05, 68.2, 72.29, 89.03]
    gpt2_xl_baseline = [11.91, 33.35, 48.55, 55.94, 59.66, 79.18]
    gpt2_xl_concat = [24.43, 47.13, 64.06, 68.23, 72.34, 88.93]
    llama2_baseline = [11.09, 24.55, 40.26, 48.56, 56.53, 74.53]
    llama2_concat = [15.06, 32.76, 51.17, 59.95, 68.45, 85.77]

    # Conll2003 sampled
    bert = [9.76, 44.39, 66.94, 72.69, 76.62, 91.18]
    bert_large = [9.29, 31.80, 53.49, 61.29, 67.53, 84.86]
    gpt2_baseline = [17.37, 37.38, 50.57, 55.59, 58.05, 77.14]
    gpt2_concat = [28.27, 46.79, 63.34, 67.42, 70.43, 88.58]
    gpt2_large_baseline = [13.89, 36.70, 49.89, 55.69, 59.64, 79.26]
    gpt2_large_concat = [27.21, 47.61, 61.42, 68.21, 71.13, 89.03]
    gpt2_xl_baseline = [18.13, 33.53, 49.14, 55.39, 60.38, 79.18]
    gpt2_xl_concat = [21.15, 47.18, 62.80, 67.62, 71.65, 88.93]
    llama2_baseline = [11.32, 24.71, 40.51, 47.48, 55.67, 74.53]
    llama2_concat = [12.05, 32.88, 49.34, 59.20, 66.97, 85.77]

    # fig = plt.figure()
    # plt = fig.add_subplot(111)

    # for i in range(5):
    #     model_size(
    #         bert, bert_large,
    #         gpt2_baseline,
    #         gpt2_large_baseline,
    #         gpt2_xl_baseline,
    #         llama2_baseline,
    #         linestyle='--',
    #         idx=i,
    #         marker='x',
    #         save=True if i == 4 else False
    #     )
    # # return

    # for i in range(5):
    #     model_size(
    #         bert, bert_large,
    #         gpt2_concat,
    #         gpt2_large_concat,
    #         gpt2_xl_concat,
    #         llama2_concat,
    #         linestyle='-',
    #         idx=i,
    #         marker='o',
    #         save=True if i == 4 else False
    #     )
    # return

    plt.rcParams["font.size"] = 14
    plt.plot(x, bert, color="orange", marker="^", label="BERT")
    plt.plot(x, bert_large, color="yellow", marker="o", label="BERT-large")
    plt.plot(
        x,
        gpt2_baseline,
        color="#191970",
        linestyle="--",
        marker="x",
        label="GPT-2 base (forward)",
    )
    plt.plot(
        x,
        gpt2_concat,
        color="#191970",
        linestyle="-",
        marker="x",
        label="GPT-2 base (forward + backward)",
    )
    plt.plot(
        x,
        gpt2_large_baseline,
        color="b",
        linestyle="--",
        marker="o",
        label="GPT2-large (baseline)",
    )
    plt.plot(
        x,
        gpt2_large_concat,
        color="b",
        linestyle="-",
        marker="o",
        label="GPT2-large (concat)",
    )
    plt.plot(
        x,
        gpt2_xl_baseline,
        color="#00bfff",
        linestyle="--",
        marker="o",
        label="GPT-2 xl (forward)",
    )
    plt.plot(
        x,
        gpt2_xl_concat,
        color="#00bfff",
        linestyle="-",
        marker="o",
        label="GPT-2 xl (forward + backward)",
    )
    plt.plot(
        x,
        llama2_baseline,
        color="green",
        linestyle="--",
        marker="o",
        label="Llama2-7b (baseline)",
    )
    plt.plot(
        x,
        llama2_concat,
        color="green",
        linestyle="-",
        marker="o",
        label="Llama2-7b (concat)",
    )

    plt.xticks(x, xlabels, fontsize=9)
    plt.xlabel("K")
    plt.ylim(0, 100)
    plt.ylabel("F1 score on CoNLL 2003")
    plt.grid(linestyle="--", alpha=0.4)
    plt.legend(prop={"size": 10})
    plt.tight_layout()
    plt.savefig("few-shot.png")
    plt.savefig("few-shot.pdf")


def model_size(
    bert,
    bert_large,
    gpt2,
    gpt2_large,
    gpt2_xl,
    llama2,
    linestyle="-",
    idx=0,
    marker="o",
    save=False,
):
    # xlabels = '109, 137, 335, 812, 1610, 7000'.split(' ')
    label_prefix = " (concat)" if marker == "o" else " (baseline)"
    print(label_prefix)
    plt.plot(
        list(map(math.log2, [109, 335])),
        [bert[idx], bert_large[idx]],
        color="orange",
        label="BERT" + label_prefix if idx == 4 else None,
        marker=marker,
        linestyle=linestyle,
        alpha=idx * 0.18 + 0.2,
    )
    plt.plot(
        list(map(math.log2, [124, 774, 1500])),
        [gpt2[idx], gpt2_large[idx], gpt2_xl[idx]],
        color="#191970",
        label="GPT-2" + label_prefix if idx == 4 else None,
        marker=marker,
        linestyle=linestyle,
        alpha=idx * 0.18 + 0.2,
    )
    plt.plot(
        list(map(math.log2, [7000])),
        [llama2[idx]],
        color="green",
        label="Llama2" + label_prefix if idx == 4 else None,
        marker=marker,
        linestyle=linestyle,
        alpha=idx * 0.18 + 0.2,
    )
    if save:
        # plt.xticks(x, xlabels, fontsize=10)
        plt.xlabel("Model Size (2^N Million)")
        plt.ylim(0, 100)
        plt.ylabel("F1 score on CoNLL 2003")
        plt.grid(linestyle="--", alpha=0.4)
        plt.legend(prop={"size": 8})
        plt.tight_layout()
        plt.savefig("few-shot.png")
        plt.savefig("few-shot.pdf")


fewshot()

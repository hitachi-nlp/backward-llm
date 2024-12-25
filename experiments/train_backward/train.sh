#!/bin/bash
#PBS -q xan_l -l select=1:res=small
#PBS -l walltime=168:00:00
#PBS -o logs/wikibook_test_seed11.o
#PBS -j oe

cd /lustre/home/71036117/bidirectional_llm/reversed_causallm
source ~/env/bin/activate
seed=11
dataset_id="wikitext-103-raw-v1"
accelerate launch \
train.py \
    --train_dataset_dir data/tokenized/$dataset_id/train/ \
    --valid_dataset_dir data/tokenized/$dataset_id/validation/ \
    --arch_id gpt2 \
    --tokenizer_id meta-llama/Llama-2-7b-chat-hf \
    --outdir models/wiki+book/seed$seed \
    --epochs 40 \
    --seed 11

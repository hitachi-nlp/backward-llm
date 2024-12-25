#!/bin/bash
#PBS -N sample
#PBS -q xvn_s -l select=1:res=small
#PBS -l walltime=24:00:00
#PBS -o logs/sample
#PBS -j oe

cd ~/bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train_baseline.py \
    --name_or_path gpt2 \
    --dataset_name conll2003 \
    --outdir models/sample2 \
    --batch_size 4 \
    --accumulation 1 \
    --epoch 20 \
    --lr 1e-3 \
    --few_shot 64

# accelerate launch train.py \
#     --name_or_path_forward meta-llama/Llama-2-7b-hf \
#     --name_or_path_backward ../backward_lm/models/wikitext+book/seed10/best/ \
#     --dataset_name conll2003 \
#     --outdir models/sample \
#     --batch_size 4 \
#     --epoch 3 \
#     --few_shot 16
for dataset in conll2003; do
for model in meta-llama/Llama-2-7b-hf; do
for fewshot in 16 64; do
for lr in 8e-3 1e-3 5e-4 1.25e-4 1e-4 5e-5 1e-5; do
for seed in 10 11 12; do
echo '#!/bin/bash
#PBS -N lb'$fewshot'_lr'$lr'
#PBS -q xvn_s -l select=1:res=small
#PBS -l walltime=24:00:00
#PBS -o logs/'$dataset'/llama2/baseline-'$fewshot'shot_seed'$seed'.txt
#PBS -j oe

cd bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train_baseline.py \
    --name_or_path '$model' \
    --dataset_name '$dataset' \
    --outdir models/'$dataset'/llama2/baseline/bsz4/'$fewshot'shot/lr'$lr'/seed'$seed' \
    --epoch 20 \
    --max_len 512 \
    --batch_size 4 \
    --accumulation 1 \
    --lr '$lr' \
    --few_shot '$fewshot' \
    --seed '$seed > temp.sh

qsub temp.sh

done
done
done
done
done
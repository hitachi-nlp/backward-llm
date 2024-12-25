for dataset in fewnerd; do
for model in gpt2-large gpt2-xl; do
for fewshot in 4 16 64 128 256; do
for lr in 1e-3; do
for seed in 10 11 12; do
echo '#!/bin/bash
#PBS -N gb'$fewshot'
#PBS -q xvn_s -l select=1:res=small
#PBS -l walltime=24:00:00
#PBS -o logs/'$dataset'/gpt2/baseline-'$fewshot'shot_seed'$seed'.txt
#PBS -j oe

cd ~/bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train_baseline.py \
    --name_or_path '$model' \
    --dataset_name '$dataset' \
    --outdir models/'$dataset'/gpt2/baseline/'$fewshot'shot/seed'$seed' \
    --epoch 20 \
    --max_len 512 \
    --batch_size 4  \
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
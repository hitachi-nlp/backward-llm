for dataset in fewnerd; do
for model in meta-llama/Llama-2-7b-hf; do
for fewshot in -1; do
for lr in 1e-3; do
for seed in 10 11 12; do
echo '#!/bin/bash
#PBS -N lb'$fewshot'_lr'$lr'
#PBS -q xan_l -l select=1:res=small
#PBS -l walltime=168:00:00
#PBS -o logs/'$dataset'/llama2/baseline-'$fewshot'shot_seed'$seed'.txt
#PBS -j oe

cd bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train_baseline.py \
    --name_or_path '$model' \
    --dataset_name '$dataset' \
    --outdir models/'$dataset'/llama2/baseline/'$fewshot'shot/seed'$seed' \
    --epoch 20 \
    --max_len 512 \
    --batch_size 8 \
    --accumulation 4 \
    --lr '$lr' \
    --few_shot '$fewshot' \
    --seed '$seed > temp.sh

qsub temp.sh

done
done
done
done
done
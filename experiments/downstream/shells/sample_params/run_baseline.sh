for dataset in conll2003; do
for model in gpt2; do
# for model in meta-llama/Llama-2-7b-hf; do
for seed in 11 12; do
echo '#!/bin/bash
#PBS -N baseline_gpt2-1000shot
#PBS -q xvn_l -l select=1:res=small
#PBS -l walltime=48:00:00
#PBS -o logs/baseline_'$dataset'_gpt2-1000shot_seed'$seed'.txt
#PBS -e logs/baseline_'$dataset'_gpt2-1000shot_seed'$seed'.txt
#PBS -j oe

cd ~/bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train_baseline.py \
    --name_or_path '$model' \
    --dataset_name '$dataset' \
    --outdir models/'$dataset'/gpt2/baseline_1000shot/seed'$seed' \
    --epoch 20 \
    --max_len 512 \
    --batch_size 32 \
    --accumulation 1 \
    --lr 1e-3 \
    --few_shot 1000
    --seed '$seed' ' > temp.sh

qsub temp.sh

done
done
done
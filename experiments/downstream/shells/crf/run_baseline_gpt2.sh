for dataset in conll2003 fewnerd; do
for model in gpt2-xl; do
for fewshot in -1; do
for lr in 1e-3; do
for seed in 10 11 12; do
mkdir -p logs/crf/$model/
echo '#!/bin/bash
#PBS -N gb'$fewshot'_crf
#PBS -q xvn_l -l select=1:res=small
#PBS -l walltime=96:00:00
#PBS -o logs/crf/'$model'/baseline_'$dataset'_'$fewshot'shot_lr'$lr'_seed'$seed'.txt
#PBS -j oe

cd ~/bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train_baseline_crf.py \
    --name_or_path '$model' \
    --dataset_name '$dataset' \
    --outdir models/'$dataset'/gpt2/baseline_crf/'$fewshot'shot/lr'$lr'/seed'$seed' \
    --epoch 20 \
    --max_len 512 \
    --batch_size 32 \
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
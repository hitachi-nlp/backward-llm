for dataset in conll2003; do
for model in gpt2; do
for fewshot in -1; do
for lr in 5e-4; do
for seed in 10 11 12; do
echo '#!/bin/bash
#PBS -N gc'$fewshot'_trans
#PBS -q xvn_s -l select=1:res=small
#PBS -l walltime=24:00:00
#PBS -o logs/trans/concat_'$dataset'_gpt2-'$fewshot'shot_lr'$lr'_seed'$seed'.txt
#PBS -j oe

cd ~/bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train_trans.py \
    --name_or_path_forward '$model' \
    --name_or_path_backward ../backward_lm/models/gpt2/wikitext/seed10/best/ \
    --dataset_name '$dataset' \
    --outdir models/'$dataset'/gpt2/concat_trans/lr'$lr'/seed'$seed' \
    --epoch 20 \
    --max_len 512 \
    --batch_size 32 \
    --accumulation 1 \
    --lr '$lr' \
    --few_shot '$fewshot' \
    --seed '$seed' ' > temp.sh

qsub temp.sh

done
done
done
done
done
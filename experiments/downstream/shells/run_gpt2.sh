for dataset in fewnerd; do
for model in gpt2-xl; do
for fewshot in -1; do
for seed in 10 11 12; do
echo '#!/bin/bash
#PBS -N gc'$fewshot'
#PBS -q xvn_l -l select=1:res=small
#PBS -l walltime=100:00:00
#PBS -o logs/'$dataset'/'$model'/concat-'$fewshot'shot_seed'$seed'.txt
#PBS -j oe

cd ~/bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train.py \
    --name_or_path_forward '$model' \
    --name_or_path_backward ../backward_lm/models/gpt2/wikitext/seed10/best/ \
    --dataset_name '$dataset' \
    --outdir models/'$dataset'/'$model'/concat/'$fewshot'shot/seed'$seed' \
    --epoch 20 \
    --max_len 512 \
    --batch_size 32 \
    --accumulation 1 \
    --lr 1e-3 \
    --few_shot '$fewshot' \
    --seed '$seed' ' > temp.sh

qsub temp.sh

done
done
done
done
for dataset in fewnerd; do
for model in gpt2 gpt2-xl; do
for fewshot in -1;do
for seed in 10 11 12; do
mkdir -p logs/random/$dataset/$model/
echo '#!/bin/bash
#PBS -N gpt-random
#PBS -q xvn_l -l select=1:res=small
#PBS -l walltime=48:00:00
#PBS -o logs/random/'$dataset'/'$model'/baseline-'$fewshot'-seed'$seed'.txt
#PBS -j oe

cd ~/bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train_random.py \
    --name_or_path_forward '$model' \
    --name_or_path_backward ../backward_lm/models/gpt2/wikitext/seed10/best/ \
    --dataset_name '$dataset' \
    --outdir models/'$dataset'/'$model'/baseline_random/'$fewshot'shot/seed'$seed' \
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
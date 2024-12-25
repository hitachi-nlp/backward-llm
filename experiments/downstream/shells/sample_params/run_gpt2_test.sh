for seed in 10 11 12; do
echo '#!/bin/bash
#PBS -N gc'$fewshot'
#PBS -q xan_s -l select=1:res=small
#PBS -l walltime=24:00:00
#PBS -o logs/gpt2_test_seed'$seed'.txt
#PBS -j oe

cd ~/bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train_test.py \
    --name_or_path_forward gpt2 \
    --name_or_path_backward ../backward_lm/models/gpt2/wikitext/seed10/best/ \
    --dataset_name conll2003 \
    --outdir models/gpt2_test/seed'$seed' \
    --epoch 20 \
    --max_len 512 \
    --batch_size 32 \
    --accumulation 1 \
    --lr 1e-3 \
    --few_shot -1 \
    --seed '$seed' ' > temp.sh

qsub temp.sh

done
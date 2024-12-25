for dataset in conll2003; do
for model in meta-llama/Llama-2-7b-hf; do
for fewshot in -1; do
for lr in 1e-3; do
for seed in 10 11 12; do
echo '#!/bin/bash
#PBS -N lc'$fewshot'_crf
#PBS -q xvn_l -l select=1:res=small
#PBS -l walltime=100:00:00
#PBS -o logs/trans/concat_'$dataset'_llama2-'$fewshot'shot_lr'$lr'_seed'$seed'.txt
#PBS -j oe

cd ~/bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train_crf.py \
    --name_or_path_forward '$model' \
    --name_or_path_backward ../backward_lm/models/wikitext+book/seed10/best/ \
    --dataset_name '$dataset' \
    --outdir models/'$dataset'/llama2/concat_crf/lr'$lr'/seed'$seed' \
    --epoch 20 \
    --max_len 512 \
    --batch_size 4 \
    --accumulation 8 \
    --lr '$lr' \
    --few_shot '$fewshot' \
    --seed '$seed > temp.sh

qsub temp.sh

done
done
done
done
done
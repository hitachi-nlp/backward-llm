for dataset in conll2003; do
for model in bert-base-cased; do
for seed in 10 11 12; do
echo '#!/bin/bash
#PBS -N bb2layer
#PBS -q xvn_s -l select=1:res=small
#PBS -l walltime=24:00:00
#PBS -o logs/crf/baseline_'$dataset'_bert-2layer_seed'$seed'.txt
#PBS -j oe

cd ~/bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train_baseline_crf.py \
    --name_or_path '$model' \
    --dataset_name '$dataset' \
    --outdir models/'$dataset'/bert-base-cased/baseline_crf/seed'$seed' \
    --epoch 20 \
    --max_len 512 \
    --batch_size 32 \
    --accumulation 1 \
    --lr 1e-3 \
    --few_shot -1 \
    --seed '$seed' ' > temp.sh

qsub temp.sh

done
done
done
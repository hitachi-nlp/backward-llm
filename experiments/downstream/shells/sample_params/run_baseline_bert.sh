for dataset in conll2003; do
for model in meta-llama/Llama-2-7b-hf; do
for fewshot in 256;do
for trial in {1..10}; do
batch_size=`python draw_param.py --json param_candidates.json --key batch_size`
lr=`python draw_param.py --json param_candidates.json --key lr`
seed=`python draw_param.py --json param_candidates.json --key seed`
dropout=`python draw_param.py --json param_candidates.json --key dropout`
echo '#!/bin/bash
#PBS -N trial'$trial'
#PBS -q xan_s -l select=1:res=small
#PBS -l walltime=24:00:00
#PBS -o logs/'$dataset'/'$model'/baseline-'$fewshot'shot-trial'$trial'.txt
#PBS -j oe

cd ~/bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train_baseline.py \
    --name_or_path '$model' \
    --dataset_name '$dataset' \
    --outdir models/'$dataset'/'$model'/baseline/'$fewshot'shot/trial'$trial' \
    --epoch 20 \
    --max_len 512 \
    --batch_size '$batch_size' \
    --accumulation 1 \
    --lr '$lr' \
    --few_shot '$fewshot' \
    --num_warmup_steps 0 \
    --dropout '$dropout' \
    --seed '$seed '
wait

cp param_candidates.json models/'$dataset'/'$model'/baseline/'$fewshot'shot/trial'$trial'/param_candidates.json
'> temp.sh


qsub temp.sh
wait

done
done
done
done
for dataset in conll2003; do
for model in meta-llama/Llama-2-7b-hf; do
for fewshot in 64; do
for trial in 2 3 4 5 6 7 8 9 10; do
batch_size=`python draw_param.py --json param_candidates.json --key batch_size`
lr=`python draw_param.py --json param_candidates.json --key lr`
seed=`python draw_param.py --json param_candidates.json --key seed`
dropout=`python draw_param.py --json param_candidates.json --key dropout`
echo '#!/bin/bash
#PBS -N trial'$trial'
#PBS -q xvn_s -l select=1:res=small
#PBS -l walltime=24:00:00
#PBS -o logs/'$dataset'/'$model'/concat-'$fewshot'shot-trial'$trial'.txt
#PBS -j oe

cd ~/bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

accelerate launch train.py \
    --name_or_path_forward '$model' \
    --name_or_path_backward ../backward_lm/models/wikitext+book/seed10/best/ \
    --dataset_name '$dataset' \
    --outdir models/'$dataset'/'$model'/concat/'$fewshot'shot/trial'$trial' \
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

cp param_candidates.json models/'$dataset'/'$model'/concat/'$fewshot'shot/trial'$trial'/param_candidates.json
' > temp.sh

qsub temp.sh
wait

done
done
done
done

# ../backward_lm/models/wikitext+book/seed10/best/
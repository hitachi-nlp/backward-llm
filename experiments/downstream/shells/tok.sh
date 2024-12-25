#!/bin/bash
#PBS -N predict
#PBS -q xan_s -l select=1:res=small
#PBS -l walltime=24:00:00
#PBS -o logs/tok.txt
#PBS -j oe

cd ~/bidirectional_llm/concatenated_lm/
source ~/.bashrc
source ~/env/bin/activate

# accelerate launch predict.py --restore_dir models/conll2003/llama2/concat/-1shot/seed10/ --batch 4
python tok.py
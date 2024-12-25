#!/bin/bash
#PBS -N sample
#PBS -q xvn_s -l select=1:res=small
#PBS -l walltime=24:00:00
#PBS -o logs/sample
#PBS -j oe

source ~/env/bin/activate
cd bidirectional_llm/concatenated_lm

accelerate launch train_baseline.py --name_or_path gpt2 --few_shot 16
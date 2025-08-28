#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=AttentionProfiler_CPU
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=24GB
#SBATCH --partition=cpu
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc

conda activate nlp-hw1

cd /export/fs06/psingh54/601.771-HW1/question1/
cp config_cpu.py config.py
python self_attention_profiler.py
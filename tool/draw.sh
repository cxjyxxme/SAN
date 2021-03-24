#!/bin/sh

## uncomment for slurm
##SBATCH -p quadro
##SBATCH --gres=gpu:1
##SBATCH -c 10

export PYTHONPATH=./
# eval "$(conda shell.bash hook)"
# conda activate pt140  # pytorch 1.4.0 env
PYTHON=python3.8

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

cp tool/draw.sh tool/draw.py ${config} ${exp_dir}

# export PYTHONPATH=./
$PYTHON -u ${exp_dir}/draw.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/draw-$now.log

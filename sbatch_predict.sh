#!/bin/bash
#SBATCH -J predict                               # 作业名为 test
#SBATCH -o slurm-%j.out                       # 屏幕上的输出文件重定向到 slurm-%j.out , %j 会替换成jobid
#SBATCH -e slurm-%j.err                       # 错误输出文件重定向到 slurm-%j.err
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                   # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=4                     # 单任务使用的 CPU 核心数为 4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:nvidia_h100:1
#SBATCH -t 0-08:00:00

source ~/anaconda3/etc/profile.d/conda.sh

conda activate llama-factory

python -m scripts.predict \
    --model_name_or_path /home/lizijian/Models/Meta-Llama-3-8B \
    --adapter_name_or_path /home/lizijian/Codes/SafeMERGE/merged_models/safealign_alpaca_cleaned_saferpaca_w_08_02 \
    --template llama3 \
    --dataset directharm4 \
    --repetition_penalty 1.2
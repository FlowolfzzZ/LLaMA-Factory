#!/bin/bash
#SBATCH -J train                               # 作业名为 test
#SBATCH -o slurm-%j.out                       # 屏幕上的输出文件重定向到 slurm-%j.out , %j 会替换成jobid
#SBATCH -e slurm-%j.err                       # 错误输出文件重定向到 slurm-%j.err
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                   # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=4                     # 单任务使用的 CPU 核心数为 4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -t 1-00:00:00

source ~/anaconda3/etc/profile.d/conda.sh

conda activate llama-factory

llamafactory-cli train examples/train_lora/llama3_lora_sft_alpaca_cleaned.yaml
# llamafactory-cli train examples/train_lora/llama3_lora_sft_health_care_magic.yaml
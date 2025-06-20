#!/bin/bash
#SBATCH -J eval                               # 作业名为 test
#SBATCH -o slurm-%j.out                       # 屏幕上的输出文件重定向到 slurm-%j.out , %j 会替换成jobid
#SBATCH -e slurm-%j.err                       # 错误输出文件重定向到 slurm-%j.err
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                   # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=4                     # 单任务使用的 CPU 核心数为 4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH -t 0-08:00:00

source ~/anaconda3/etc/profile.d/conda.sh

conda activate llama-factory

tasks="ifeval"

lm_eval --model hf \
    --model_args pretrained=/home/lizijian/Models/Meta-Llama-3-8B,peft=/home/lizijian/Loras/Meta-Llama-3-8B/saferpaca \
    --tasks ${tasks} \
    --num_fewshot 0 \
    --device cuda:0 \
    --batch_size 16 \
    --output_path="./saves/Meta-Llama-3-8B-${tasks}"
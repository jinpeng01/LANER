#!/bin/bash
#SBATCH -J 1_2
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
###########################

python main.py \
--exp_name ai_experiment \
--exp_id ai_experiment \
--num_tag 29 \
--batch_size 16 \
--ckpt /mntnfs/diis_data3/chenqian/workspace/CrossNER_pre_trained/ai_spanlevel_integrated/pytorch_model.bin \
--tgt_dm ai \
--target_sequence \
--seed 8888 \
--target_embedding_dim 100 \
--target_type RNN \
--connect_label_background \
--conll





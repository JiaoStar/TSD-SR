#!/usr/bin/env bash
cd "$(dirname "$0")"

gpu_id="${gpu_id:-0}"

# Yelp
python run_model.py --gpu_id=${gpu_id} --model=tsdsr --dataset="yelp" --config_files="configs/yelp_tsdsr.yaml" --fusion_type='gate' --alpha=0.3 --lambda=20 --M=16 --reg_weight=1e-3

# Beauty
python run_model.py --gpu_id=${gpu_id} --model=tsdsr --dataset="Amazon_Beauty" --config_files="configs/Amazon_Beauty_tsdsr.yaml" --fusion_type='concat' --alpha=0.3 --lambda=100 --lr=5e-4 --M=32 --reg_weight=0

# Sports
python run_model.py --gpu_id=${gpu_id} --model=tsdsr --dataset="Amazon_Sports_and_Outdoors" --config_files="configs/Amazon_Sports_and_Outdoors_tsdsr.yaml" --fusion_type='concat' --alpha=0.4 --lambda=100 --lr=5e-4 --M=48 --reg_weight=0 --K=50

# Toys
python run_model.py --gpu_id=${gpu_id} --model=tsdsr --dataset="Amazon_Toys_and_Games" --config_files="configs/Amazon_Toys_and_Games_tsdsr.yaml" --fusion_type='concat' --alpha=0.38 --lambda=100

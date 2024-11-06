#!/bin/bash
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.4 --sparsity_type unstructured --save out/llama2_7b/u_40 --save_model out/llama2_7b/u_40
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.3 --sparsity_type unstructured --save out/llama2_7b/u_30 --save_model out/llama2_7b/u_30
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.2 --sparsity_type unstructured --save out/llama2_7b/u_20 --save_model out/llama2_7b/u_20
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.1 --sparsity_type unstructured --save out/llama2_7b/u_10 --save_model out/llama2_7b/u_10
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.6 --sparsity_type unstructured --save out/llama2_7b/u_60 --save_model out/llama2_7b/u_60
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.7 --sparsity_type unstructured --save out/llama2_7b/u_70 --save_model out/llama2_7b/u_70
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.8 --sparsity_type unstructured --save out/llama2_7b/u_80 --save_model out/llama2_7b/u_80
python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.9 --sparsity_type unstructured --save out/llama2_7b/u_90 --save_model out/llama2_7b/u_90

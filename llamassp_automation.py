import os
import argparse
import logging
from lssp.base import create_model
from lssp.base import sample_model
from lssp import evals
from lssp.ssp import ssp
import sys
import time
import torch
from transformers import AutoTokenizer
from termcolor import colored
torch.manual_seed(42)

##########모델 불러온 다음 quantization하기##########

import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant_2_bits import quantize_model as quantize_model_2
from smoothquant.fake_quant_4_bits import quantize_model as quantize_model_4
from smoothquant.fake_quant_6_bits import quantize_model as quantize_model_6
from smoothquant.fake_quant_8_bits import quantize_model as quantize_model_8
import tqdm

from datasets import load_dataset

alpha = 0.5
ssm_model_path = 'facebook/opt-1.3b'
ssm_act_scales_path = 'act_scales/opt-1.3b.pt'
ltm_model_path = 'facebook/opt-6.7b'
ltm_act_scales_path = 'act_scales/opt-6.7b.pt'
n_samples = None

# Argument parsing and setup
def parse_args():
    parser = argparse.ArgumentParser(description='Test speeds of Llama models with regular sampling and speculative sampling.')
    parser.add_argument('--ssm_bit', choices=['2', '4', '6', '8', 'none'], default='none', help="Quantization bit for SSM model")
    parser.add_argument('--ltm_bit', choices=['2', '4', '6', '8', 'none'], default='none', help="Quantization bit for LTM model")
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha value for smooth quantization")
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    return parser.parse_args()

# Load and quantize models based on arguments
def load_model(model_path, act_scales_path, bit, alpha):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    act_scales = torch.load(act_scales_path)
    smooth_lm(model, act_scales, alpha)

    if bit == '2':
        model = quantize_model_2(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True)
    elif bit == '4':
        model = quantize_model_4(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True)
    elif bit == '6':
        model = quantize_model_6(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True)
    elif bit == '8':
        model = quantize_model_8(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True)
    
    return model

# Model paths for both SSM and LTM models
ssm_model = load_model(ssm_model_path, ssm_act_scales_path, 'none', alpha)
ltm_model = load_model(ltm_model_path, ltm_act_scales_path, 'none', alpha)

MAX_NEW_TOKENS = 64
llama7b_name = 'facebook/opt-1.3b'
llama13b_name = 'facebook/opt-6.7b'
llama30b_name = 'baffo32/decapoda-research-llama-30b-hf'
llama65b_name = 'meta-llama/Llama-2-70b-hf'
batch_size = 1

dataset = load_dataset("lambada", split="test")  # LAMBADA 데이터셋
texts = dataset["text"][:100]  # 첫 100개의 텍스트를 사용

tokenizer = AutoTokenizer.from_pretrained(llama7b_name)

free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
max_mem = f'{int(torch.cuda.mem_get_info()[0] / 1024**3) - 2}GB'

n_gpus = torch.cuda.device_count()

def max_memory(gpus, starting_gpu=0):
    return {i: max_mem for i in range(starting_gpu, n_gpus)}

# Modify the time_ssp function to use provided draft and ltm models
def time_ssp(target_name, draft_name, draft, ltm, K=4):
    draft_model = draft
    target_model = ltm
    nb_tokens = 0
    all_accept_tokens = 0
    all_generated_tokens = 0

    # 전체 실행 시간 측정을 위한 초기화
    start_time = time.time()

    input_ids = tokenizer(texts[0], return_tensors="pt").input_ids
    input_ids = torch.stack([input_ids[0]] * batch_size).to(draft_model.device)

    # 첫 번째 텍스트 처리
    print("Generating tokens for:", tokenizer.decode(input_ids[0]))
    print("=" * 40)
    generated_ids, accept_tokens, generated_tokens = ssp(
        target_model,
        draft_model,
        MAX_NEW_TOKENS,
        input_ids,
        K=K,
        display=True
    )
    print("Completion:", tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    print("Acceptance Rate: {:.2f}%".format((accept_tokens / generated_tokens) * 100))
    print("Total Time: {:.2f}s".format(time.time() - start_time))
    print("=" * 40)

    start_time = time.time()  # 전체 실행 시간 재설정
    for text in texts[1:]:
        print("Generating tokens for:", text)
        print("=" * 40)

        # 텍스트별 생성 시간 측정을 위한 시간 초기화
        intermediate_time = time.time()

        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = torch.stack([input_ids[0]] * batch_size).to(draft_model.device)

        # Speculative Sampling 수행
        generated_ids, accept_tokens, generated_tokens = ssp(
            target_model,
            draft_model,
            MAX_NEW_TOKENS,
            input_ids,
            K=K,
            display=True
        )

        all_accept_tokens += accept_tokens
        all_generated_tokens += generated_tokens
        nb_tokens += generated_ids.shape[1] - input_ids.shape[1]

        print("Completion:", tokenizer.decode(generated_ids[0], skip_special_tokens=True))
        print("Acceptance Rate: {:.2f}%".format((accept_tokens / generated_tokens) * 100))
        print("Time for this text: {:.2f}s".format(time.time() - intermediate_time))
        print("=" * 40)

    ms_per_token = (time.time() - start_time) * 1000 / nb_tokens
    accept_rate = (all_accept_tokens / all_generated_tokens)
    return generated_ids, ms_per_token, accept_rate

# Main function that processes args
if __name__ == "__main__":
    args = parse_args()

    # Set verbose logging if specified
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Load models based on quantization args
    ssm_model = load_model(ssm_model_path, ssm_act_scales_path, args.ssm_bit, args.alpha)
    ltm_model = load_model(ltm_model_path, ltm_act_scales_path, args.ltm_bit, args.alpha)

    # Example function call, modify as needed
    tokens_s, outputs, accept_rate = time_ssp('7B', '65B_8bit', ssm_model, ltm_model)
    print(f"Results: ms per token: {tokens_s:.2f}, acceptance rate: {accept_rate * 100:.2f}%")

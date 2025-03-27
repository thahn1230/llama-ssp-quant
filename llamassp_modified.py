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
from transformers import LlamaTokenizer, GPT2Tokenizer
from termcolor import colored
torch.manual_seed(42)

##########모델 불러온 다음 quantization하기##########

import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
# from smoothquant.smooth import smooth_lm
# from smoothquant.fake_quant_2_bits import quantize_model as quantize_model_2
# from smoothquant.fake_quant_4_bits import quantize_model as quantize_model_4
# from smoothquant.fake_quant_6_bits import quantize_model as quantize_model_6
# from smoothquant.fake_quant_8_bits import quantize_model as quantize_model_8
# import tqdm

from datasets import load_dataset

# alpha = 0.5
# ssm_model_path = 'facebook/opt-125m'
# ssm_act_scales_path = 'act_scales/opt-1.3b.pt'
# ltm_model_path = 'facebook/opt-6.7b'
# ltm_act_scales_path = 'act_scales/opt-6.7b.pt'
# n_samples = None

# ssmmodel = AutoModelForCausalLM.from_pretrained(
#     ssm_model_path, torch_dtype=torch.bfloat16, device_map="auto"
# )

# # smooth
# ssm_act_scales = torch.load(ssm_act_scales_path)
# smooth_lm(ssmmodel, ssm_act_scales, alpha)

# # quantize
# ssmmodel = quantize_model_6(
#     ssmmodel,
#     weight_quant="per_channel",
#     act_quant="per_token",
#     quantize_bmm_input=True,
# )

# ltmmodel = AutoModelForCausalLM.from_pretrained(
#     ltm_model_path, torch_dtype=torch.bfloat16, device_map="auto"
# )

# # smooth
# ltm_act_scales = torch.load(ltm_act_scales_path)
# smooth_lm(ltmmodel, ltm_act_scales, alpha)

# # quantize
# ltmmodel = quantize_model_6(
#     ltmmodel,
#     weight_quant="per_channel",
#     act_quant="per_token",
#     quantize_bmm_input=True,
# )

# Load and quantize models based on arguments
def load_model(model_path, act_scales_path, bit, alpha):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    act_scales = torch.load(act_scales_path)
    smooth_lm(model, act_scales, alpha)

    if bit == 2:
        model = quantize_model_2(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True)
    elif bit == 4:
        model = quantize_model_4(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True)
    elif bit == 6:
        model = quantize_model_6(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True)
    elif bit == 8:
        model = quantize_model_8(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True)
    elif bit == 0:
        return model
    
    return model

################################################
MAX_NEW_TOKENS = 64
llama7b_name = 'facebook/opt-125m'
llama13b_name = 'facebook/opt-6.7b'
llama30b_name = 'baffo32/decapoda-research-llama-30b-hf'
llama65b_name = 'meta-llama/Llama-2-70b-hf'
batch_size = 1

dataset = load_dataset("lambada", split="test")  # LAMBADA 데이터셋
texts = dataset["text"][:100]  # 첫 100개의 텍스트를 사용

tokenizer = AutoTokenizer.from_pretrained(llama7b_name)

free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
max_mem = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

n_gpus = torch.cuda.device_count()


def max_memory(gpus, starting_gpu=0):
    return {i: max_mem for i in range(starting_gpu, n_gpus)}


def time_model(model):
    # time the first run
    input_ids = tokenizer(texts[0], return_tensors="pt").input_ids
    input_ids = torch.stack([input_ids[0]] * batch_size).to(model.device)
    generated_ids = sample_model(model, input_ids, MAX_NEW_TOKENS)

    start_time = time.time()
    nb_tokens = 0
    for text in texts[1:]:
        print("Completing text:", text)
        intermediate_time = time.time()
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = torch.stack([input_ids[0]] * batch_size).to(model.device)
        generated_ids = sample_model(model, input_ids, MAX_NEW_TOKENS)
        nb_tokens += generated_ids.shape[1] - input_ids.shape[1]
        print("Completion: ", tokenizer.decode(
            generated_ids[0], skip_special_tokens=True))
        print("Time: {:.2f}s".format(time.time() - intermediate_time))
        print("========\n")
    ms_per_token = (time.time() - start_time)*1000 / nb_tokens
    return generated_ids, ms_per_token


def print_results(tokens_s, outputs, accept_rate, name='Noname'):
    print("Results for ", name)
    print(f"Ms per token: {tokens_s:.2f}ms")
    print("========\n")
    print(f"Total accecptance rate: {accept_rate*100:.2f}%")
    print("========\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("========\n")


models_params = {
    '7B_8bit': {'model_name': llama7b_name,
                'max_memory': max_memory(1),
                'load_in_8bit': True},
    '7B_8bit_4': {'model_name': llama7b_name,
                  'max_memory': max_memory(4),
                  'load_in_8bit': True},
    '7B': {'model_name': llama7b_name,
           'max_memory': max_memory(1),
           'load_in_8bit': False},
    '7B_8': {'model_name': llama7b_name,
             'max_memory': max_memory(8),
             'load_in_8bit': False},
    '13B_8bit': {'model_name': llama13b_name,
                 'max_memory': max_memory(1),
                 'load_in_8bit': True},
    '13B': {'model_name': llama13b_name,
            'max_memory': max_memory(2),
            'load_in_8bit': False},
    '30B_8bit': {'model_name': llama30b_name,
                 'max_memory': max_memory(2),
                 'load_in_8bit': True},
    '30B': {'model_name': llama30b_name,
            'max_memory': max_memory(4),
            'load_in_8bit': False},
    '65B_8bit': {'model_name': llama65b_name,
                 'max_memory': max_memory(4),
                 'load_in_8bit': True},
    '65B': {'model_name': llama65b_name,
            'max_memory': max_memory(8),
            'load_in_8bit': False},
    '65B_v2': {'model_name': f"{os.getenv('HOME')}/data/hf-weights/65B",
               'max_memory': max_memory(8),
               'load_in_8bit': False},
}


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


def print_speeds(speeds):
    print("Speeds:")
    for model_name, tokens_s in speeds.items():
        print('-'*20)
        print(f"{model_name} |  {tokens_s:.2f}ms")
    print('-'*20)


def models_raw_speed():
    speeds = {}
    del models_params['7B'], models_params['13B'], models_params['30B']
    for model_name, params in sorted(models_params.items()):
        print(f"Testing {model_name}")
        print('-'*20)
        model = create_model(**params)
        outputs, tokens_s = time_model(model)
        speeds[model_name] = tokens_s
        print_results(tokens_s, outputs, model_name)
        del model
        torch.cuda.empty_cache()
        print_speeds(speeds)
    draft_name = '7B_8bit'
    target_name = '65B_8bit'
    print(f"Testing SSP {draft_name} / {target_name}")
    tokens_s, outputs, accept_rate = time_ssp(draft_name, target_name, draft=ssmmodel, ltm=ltmmodel)
    speeds[f"{draft_name} / {target_name}"] = tokens_s
    print(speeds)


def show_comparative_speeds(text, model, draft_model):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(colored("=> Regular sampling with target model",
                  attrs=['bold']))
    sys.stdout.write(text)
    start_time = time.time()
    sample_model(model, input_ids, MAX_NEW_TOKENS, display=True)
    print("\nTime: "
          + colored(f"{time.time() - start_time:.2f}s", 'red', attrs=['bold']))
    print(colored(
        "=> Speculative sampling with target model helped by draft model",
        attrs=['bold']))
    sys.stdout.write(text)
    start_time = time.time()
    ssp(model, draft_model, MAX_NEW_TOKENS,
        input_ids, K=4, display=True)
    print("\nTime: "
          + colored(f"{time.time() - start_time:.2f}s", 'green', attrs=['bold']))


def create_argument_parser():
    """
    Create a parser for the command-line arguments, with 'compare', 'latency'
    and 'eval' subcommands
    """
    parser = argparse.ArgumentParser(
        description='Test speeds of Llama models with regular sampling and speculative sampling: measure their latency, compare their speed, and evaluate their performance on a simple task.')
    # my arg
    parser.add_argument('--ssm-bit', type=int, default=0, help="Quantization bit for SSM model")
    parser.add_argument('--ltm-bit', type=int, default=0, help="Quantization bit for LTM model")
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha value for smooth quantization")
    
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='verbose output')

    subparsers = parser.add_subparsers(dest='subcommand')
    compare_parser = subparsers.add_parser(
        'compare', help='Compare the speed of a given model (target model) alone, and with speculative sampling with another model (draft model)')
    compare_parser.add_argument('model', help='Name of target model')
    compare_parser.add_argument('draft', help='Draft model')

    latency_parser = subparsers.add_parser(
        'latency', help='Measure model latency in ms per token')
    latency_parser.add_argument('model', help='Name of model')
    latency_parser.add_argument(
        '--draft', help='Draft model; if specified, will measure the latency of speculative sampling with the draft model rather than the regular latency')

    eval_parser = subparsers.add_parser(
        'eval', help='evaluate a model')
    eval_parser.add_argument('model', help='model to use')
    eval_parser.add_argument(
        '--draft', help='Draft model; if specified, will evaluate the model with speculative sampling with the draft model rather than the regular model')
    eval_parser.add_argument('--seed', type=int, default=1338,
                             help='Seed for randomly creating the eval prompts')
    eval_parser.add_argument('--nb-prompts', type=int, default=1000,
                             help='Number of eval prompts to create')
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        # set log level to debug
        logging.basicConfig(level=logging.DEBUG)

    if args.subcommand == 'compare':
        model = create_model(**models_params[args.model])
        draft_model = create_model(**models_params[args.draft])
        print("Warming up")
        ssp(model, draft_model, MAX_NEW_TOKENS,
            tokenizer(texts[0], return_tensors="pt").input_ids, K=4)
        print(
            f"Comparing {args.model} model regular sampling and {args.model} SSp with {args.draft} draft model\n====\n")
        # Read from stdin until EOF
        while True:
            try:
                sys.stdout.write("> ")
                sys.stdout.flush()
                text = input()
            except EOFError:
                break
            show_comparative_speeds(text, model, draft_model)

    elif (args.subcommand == 'latency' and args.draft):
        print(f"Testing {args.model} with draft {args.draft}")
        print('-'*20)
        # Load models based on quantization args
        if args.ssm_bit == 0:
            ssm_model = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-1.3b",
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
        else:
            ssm_model = load_model(ssm_model_path, ssm_act_scales_path, args.ssm_bit, args.alpha)
        
        if args.ltm_bit == 0:
            ltm_model = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-6.7b",
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
        else:
            ltm_model = load_model(ltm_model_path, ltm_act_scales_path, args.ltm_bit, args.alpha)
        gen_ids, ms_per_token, accept_rate = time_ssp(args.model, args.draft, ssm_model, ltm_model)
        print_results(ms_per_token, gen_ids, accept_rate, args.model)

    elif (args.subcommand == 'latency'):
        print(f"Testing {args.model}")
        print('-'*20)
        model = create_model(**models_params[args.model])
        gen_ids, ms_per_token = time_model(model)
        accept_rate = None
        print_results(ms_per_token, gen_ids, accept_rate, args.model)

    elif (args.subcommand == 'eval'):
        print(f"Eval of {args.model} on multiplication task (seed {args.seed})"
              + (f" with draft {args.draft}" if args.draft else ""))
        print('-'*20)
        model = create_model(**models_params[args.model])
        if args.draft:
            draft_model = create_model(**models_params[args.draft])
        else:
            draft_model = None
        results = evals.measure_model_score(
            model, tokenizer, args.nb_prompts, args.seed, draft_model)
        evals.print_results(results, args.model, args.draft)

    else:
        # show usage
        parser.print_help()

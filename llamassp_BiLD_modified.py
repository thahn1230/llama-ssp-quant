import os
import argparse
import logging
from lssp.base import create_model
from lssp.base import sample_model
from lssp import evals
from lssp.ssp_modified import ssp  # ssp_modified에서 새로운 ssp 함수 사용
import sys
import time
import torch
from transformers import AutoTokenizer
from termcolor import colored
torch.manual_seed(42)

from datasets import load_dataset


MAX_NEW_TOKENS = 64
llama7b_name = 'facebook/opt-125m'
llama13b_name = 'facebook/opt-6.7b'
llama30b_name = 'baffo32/decapoda-research-llama-30b-hf'
llama65b_name = 'meta-llama/Llama-2-70b-hf'
batch_size = 1

# wikitext-2 데이터셋 사용
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = dataset["text"][:100]

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


def print_results(tokens_s, outputs, accept_rate, time_stats=None, name='Noname'):
    print("Results for ", name)
    print(f"Ms per token: {tokens_s:.2f}ms")
    print("========\n")
    
    if accept_rate is not None:
        print(f"Total acceptance rate: {accept_rate*100:.2f}%")
        print("========\n")
    
    # 시간 통계 출력 추가
    if time_stats:
        print("Time breakdown:")
        total_time = sum(time_stats.values())
        for key, value in time_stats.items():
            percentage = (value / total_time) * 100 if total_time > 0 else 0
            print(f"- {key}: {value:.4f}s ({percentage:.2f}%)")
        
        # 초당 토큰 수(TPS) 계산
        if tokens_s > 0:
            tps = 1000 / tokens_s
            print(f"Tokens per second: {tps:.2f}")
        print("========\n")
    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("========\n")


models_params = {
    '7B_8bit': {'model_name': llama7b_name,
                'max_memory': max_memory(1)},
    '7B_8bit_4': {'model_name': llama7b_name,
                  'max_memory': max_memory(4)},
    '7B': {'model_name': llama7b_name,
           'max_memory': max_memory(1)},
    '7B_8': {'model_name': llama7b_name,
             'max_memory': max_memory(8)},
    '13B_8bit': {'model_name': llama13b_name,
                 'max_memory': max_memory(1)},
    '13B': {'model_name': llama13b_name,
            'max_memory': max_memory(2)},
    '30B_8bit': {'model_name': llama30b_name,
                 'max_memory': max_memory(2)},
    '30B': {'model_name': llama30b_name,
            'max_memory': max_memory(4)},
    '65B_8bit': {'model_name': llama65b_name,
                 'max_memory': max_memory(4)},
    '65B': {'model_name': llama65b_name,
            'max_memory': max_memory(8)},
    '65B_v2': {'model_name': f"{os.getenv('HOME')}/data/hf-weights/65B",
               'max_memory': max_memory(8)},
}


def time_ssp(target_name, draft_name, K=16, fallback_threshold=None, rollback_threshold=None):
    # 7B-7B 조합은 스킵
    if (target_name.startswith("7B") and draft_name.startswith("7B")):
        print(f"Skipping 7B-7B combination ({target_name}-{draft_name}).")
        # Return dummy values for compatibility
        return None, 0.0, 0.0, {
            'draft_generation': 0,
            'target_verification': 0,
            'fallback_handling': 0,
            'rollback_handling': 0,
            'token_acceptance': 0,
            'overhead': 0
        }
        
    # 모델 로딩 시간 측정
    model_load_start = time.time()
    draft_model = create_model(**models_params[draft_name])
    draft_load_time = time.time() - model_load_start
    
    target_load_start = time.time()
    target_model = create_model(**models_params[target_name])
    target_load_time = time.time() - target_load_start
    
    print(f"Model loading times - Draft: {draft_load_time:.2f}s, Target: {target_load_time:.2f}s")

    nb_tokens = 0
    # Warmup
    input_ids = tokenizer(texts[0], return_tensors="pt").input_ids
    input_ids = torch.stack(
        [input_ids[0]] * batch_size).to(draft_model.device)
    # 워밍업에는 기본 시간 통계를 무시
    generated_ids, accept_tokens, generated_tokens, _ = ssp(target_model,
                        draft_model,
                        MAX_NEW_TOKENS,
                        input_ids, K=K,
                        fallback_threshold=fallback_threshold,
                        rollback_threshold=rollback_threshold)

    start_time = time.time()
    all_accept_tokens = 0
    all_generated_tokens = 0
    
    # 각 단계별 누적 시간 통계
    all_time_stats = {
        'draft_generation': 0,
        'target_verification': 0,
        'fallback_handling': 0,
        'rollback_handling': 0,
        'token_acceptance': 0,
        'overhead': 0
    }
    
    for text in texts[1:]:
        print("Completing text:", text)
        intermediate_time = time.time()
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = torch.stack(
            [input_ids[0]] * batch_size).to(draft_model.device)
        
        # 수정된 ssp 함수 호출 (시간 통계 반환)
        generated_ids, accept_tokens, generated_tokens, time_stats = ssp(target_model,
                            draft_model,
                            MAX_NEW_TOKENS,
                            input_ids, K=K,
                            fallback_threshold=fallback_threshold,
                            rollback_threshold=rollback_threshold)
        
        # 시간 통계 누적
        for key in time_stats:
            all_time_stats[key] += time_stats[key]
        
        all_accept_tokens += accept_tokens
        all_generated_tokens += generated_tokens
        nb_tokens += generated_ids.shape[1] - input_ids.shape[1]
        
        print("Completion: ", tokenizer.decode(
            generated_ids[0], skip_special_tokens=True))
        print("Time: {:.2f}s".format(time.time() - intermediate_time))
        print("========\n")
        print("Acceptance Rate: {:.2f}%".format((accept_tokens/generated_tokens)*100 if generated_tokens > 0 else 0))
        
        # 현재 텍스트에 대한 시간 통계 출력
        print("Time Breakdown for this text:")
        total_time = sum(time_stats.values())
        for key, value in time_stats.items():
            percentage = (value / total_time) * 100 if total_time > 0 else 0
            print(f"- {key}: {value:.4f}s ({percentage:.2f}%)")
        print("========\n")
    
    total_time = time.time() - start_time
    ms_per_token = total_time * 1000 / nb_tokens if nb_tokens > 0 else 0
    accept_rate = (all_accept_tokens / all_generated_tokens) if all_generated_tokens > 0 else 0
    
    return generated_ids, ms_per_token, accept_rate, all_time_stats


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
        print_results(tokens_s, outputs, None, None, model_name)
        del model
        torch.cuda.empty_cache()
        print_speeds(speeds)
    draft_name = '7B_8bit'
    target_name = '65B_8bit'
    print(f"Testing SSP {draft_name} / {target_name}")
    tokens_s, outputs, accept_rate, time_stats = time_ssp(draft_name, target_name)
    speeds[f"{draft_name} / {target_name}"] = tokens_s
    print_speeds(speeds)
    print_results(tokens_s, outputs, accept_rate, time_stats)


def show_comparative_speeds(text, model, draft_model, fallback_threshold=None, rollback_threshold=None):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(colored("=> Regular sampling with target model",
                  attrs=['bold']))
    sys.stdout.write(text)
    start_time = time.time()
    sample_model(model, input_ids, MAX_NEW_TOKENS, display=True)
    regular_time = time.time() - start_time
    print("\nTime: "
          + colored(f"{regular_time:.2f}s", 'red', attrs=['bold']))
          
    print(colored(
        "=> Speculative sampling with target model helped by draft model",
        attrs=['bold']))
    sys.stdout.write(text)
    start_time = time.time()
    _, accept_tokens, generated_tokens, time_stats = ssp(model, draft_model, MAX_NEW_TOKENS,
        input_ids, K=16, display=True,
        fallback_threshold=fallback_threshold,
        rollback_threshold=rollback_threshold)
    ssp_time = time.time() - start_time
    print("\nTime: "
          + colored(f"{ssp_time:.2f}s", 'green', attrs=['bold']))
    
    # 속도 향상 및 수락률 계산
    speedup = regular_time / ssp_time if ssp_time > 0 else 0
    accept_rate = accept_tokens / generated_tokens if generated_tokens > 0 else 0
    
    print("\nPerformance comparison:")
    print(f"- Speedup: {speedup:.2f}x")
    print(f"- Acceptance rate: {accept_rate*100:.2f}%")
    
    # 시간 통계 출력
    print("\nSSP Time breakdown:")
    total_time = sum(time_stats.values())
    for key, value in time_stats.items():
        percentage = (value / total_time) * 100 if total_time > 0 else 0
        print(f"- {key}: {value:.4f}s ({percentage:.2f}%)")


def create_argument_parser():
    """
    Create a parser for the command-line arguments, with 'compare', 'latency'
    and 'eval' subcommands
    """
    parser = argparse.ArgumentParser(
        description='Test speeds of Llama models with regular sampling and speculative sampling: measure their latency, compare their speed, and evaluate their performance on a simple task.')
    # add argument to set log level
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='verbose output')

    subparsers = parser.add_subparsers(dest='subcommand')
    compare_parser = subparsers.add_parser(
        'compare', help='Compare the speed of a given model (target model) alone, and with speculative sampling with another model (draft model)')
    compare_parser.add_argument('model', help='Name of target model')
    compare_parser.add_argument('draft', help='Draft model')
    compare_parser.add_argument('--fallback-threshold', type=float, default=None,
                               help='Threshold for small model fallback policy (0-1.0)')
    compare_parser.add_argument('--rollback-threshold', type=float, default=None,
                               help='Threshold for rollback policy based on cross-entropy distance')

    latency_parser = subparsers.add_parser(
        'latency', help='Measure model latency in ms per token')
    latency_parser.add_argument('model', help='Name of model')
    latency_parser.add_argument(
        '--draft', help='Draft model; if specified, will measure the latency of speculative sampling with the draft model rather than the regular latency')
    latency_parser.add_argument('--fallback-threshold', type=float, default=None,
                               help='Threshold for small model fallback policy (0-1.0)')
    latency_parser.add_argument('--rollback-threshold', type=float, default=None,
                               help='Threshold for rollback policy based on cross-entropy distance')

    eval_parser = subparsers.add_parser(
        'eval', help='evaluate a model')
    eval_parser.add_argument('model', help='model to use')
    eval_parser.add_argument(
        '--draft', help='Draft model; if specified, will evaluate the model with speculative sampling with the draft model rather than the regular model')
    eval_parser.add_argument('--seed', type=int, default=1338,
                             help='Seed for randomly creating the eval prompts')
    eval_parser.add_argument('--nb-prompts', type=int, default=1000,
                             help='Number of eval prompts to create')
    eval_parser.add_argument('--fallback-threshold', type=float, default=None,
                           help='Threshold for small model fallback policy (0-1.0)')
    eval_parser.add_argument('--rollback-threshold', type=float, default=None,
                           help='Threshold for rollback policy based on cross-entropy distance')
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        # set log level to debug
        logging.basicConfig(level=logging.DEBUG)

    if args.subcommand == 'compare':
        # 7B-7B 조합은 스킵
        if (args.model.startswith("7B") and args.draft.startswith("7B")):
            print(f"Skipping 7B-7B combination ({args.model}-{args.draft}).")
            sys.exit(0)
            
        model = create_model(**models_params[args.model])
        draft_model = create_model(**models_params[args.draft])
        print("Warming up")
        ssp(model, draft_model, MAX_NEW_TOKENS,
            tokenizer(texts[0], return_tensors="pt").input_ids, K=16,
            fallback_threshold=args.fallback_threshold,
            rollback_threshold=args.rollback_threshold)
        policies_info = ""
        if args.fallback_threshold or args.rollback_threshold:
            policies_info = f" (fallback threshold: {args.fallback_threshold}, rollback threshold: {args.rollback_threshold})"
        print(
            f"Comparing {args.model} model regular sampling and {args.model} SSp with {args.draft} draft model{policies_info}\n====\n")
        # Read from stdin until EOF
        while True:
            try:
                sys.stdout.write("> ")
                sys.stdout.flush()
                text = input()
            except EOFError:
                break
            show_comparative_speeds(text, model, draft_model, 
                                   fallback_threshold=args.fallback_threshold,
                                   rollback_threshold=args.rollback_threshold)

    elif (args.subcommand == 'latency' and args.draft):
        # 7B-7B 조합은 스킵
        if (args.model.startswith("7B") and args.draft.startswith("7B")):
            print(f"Skipping 7B-7B combination ({args.model}-{args.draft}).")
            sys.exit(0)
            
        print(f"Testing {args.model} with draft {args.draft}")
        print('-'*20)
        if args.fallback_threshold or args.rollback_threshold:
            print(f"Using fallback threshold: {args.fallback_threshold}, rollback threshold: {args.rollback_threshold}")
        gen_ids, ms_per_token, accept_rate, time_stats = time_ssp(
            args.model, args.draft,
            fallback_threshold=args.fallback_threshold,
            rollback_threshold=args.rollback_threshold
        )
        print_results(ms_per_token, gen_ids, accept_rate, time_stats, args.model)

    elif (args.subcommand == 'latency'):
        print(f"Testing {args.model}")
        print('-'*20)
        model = create_model(**models_params[args.model])
        gen_ids, ms_per_token = time_model(model)
        accept_rate = None
        print_results(ms_per_token, gen_ids, accept_rate, None, args.model)

    elif (args.subcommand == 'eval'):
        # 7B-7B 조합은 스킵
        if args.draft and args.model.startswith("7B") and args.draft.startswith("7B"):
            print(f"Skipping 7B-7B combination ({args.model}-{args.draft}).")
            sys.exit(0)
            
        print(f"Eval of {args.model} on multiplication task (seed {args.seed})"
              + (f" with draft {args.draft}" if args.draft else ""))
        print('-'*20)
        if args.fallback_threshold or args.rollback_threshold:
            print(f"Using fallback threshold: {args.fallback_threshold}, rollback threshold: {args.rollback_threshold}")
        model = create_model(**models_params[args.model])
        if args.draft:
            draft_model = create_model(**models_params[args.draft])
        else:
            draft_model = None
        results = evals.measure_model_score(
            model, tokenizer, args.nb_prompts, args.seed, draft_model,
            fallback_threshold=args.fallback_threshold,
            rollback_threshold=args.rollback_threshold)
        evals.print_results(results, args.model, args.draft)

    else:
        # show usage
        parser.print_help() 
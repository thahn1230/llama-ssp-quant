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
from transformers import LlamaTokenizer
from termcolor import colored
torch.manual_seed(42)

MAX_NEW_TOKENS = 64
llama7b_name = '/home/sangjun/wanda/out/llama2_7b/u_50'
llama13b_name = 'meta-llama/Llama-2-13b-hf'
llama30b_name = 'baffo32/decapoda-research-llama-30b-hf'
llama65b_name = 'meta-llama/Llama-2-70b-hf'
batch_size = 1

texts = [
    'In which country is Hamburg?\n',
    'How are you doing today?\n',
    'It was a dark and stormy night.',
    'The sun rose slowly over the horizon, casting a warm glow on the world below.',
    'I never believed in ghosts until the day I met one.',
    'The sound of the train whistle echoed through the valley as I stood at the station, waiting.',
    'She walked into the room and everything changed.',
    'The smell of freshly baked bread filled the air as I entered the bakery.',
    'The first time I saw her, I knew she was trouble.'
    'The world was ending, and I was the only one who knew.',
    'It was the best of times, it was the worst of times.',
    'The forest was alive with the sound of animals as I walked deeper into the woods.',
    'As I looked out over the city, I knew that anything was possible.',
    'The sound of gunfire echoed through the streets as I ran for cover.',
    'The waves crashed against the shore, a never-ending cycle of destruction and creation.',
    'I woke up to find myself in a strange place, with no memory of how I got there.',
    'The clock struck midnight, and I knew that my life would never be the same.',
    'What country is Berlin located in?',
    'How are you feeling this morning?',
    'The wind howled through the trees on a cold, wintry night.',
    'The stars began to fade as the first light of dawn appeared on the horizon.',
    'I used to laugh at the idea of aliens, until I saw one myself.',
    'The distant sound of church bells echoed through the quiet village.',
    'As soon as he entered the room, the atmosphere shifted.',
    'The aroma of roasted coffee beans filled the small café as I opened the door.',
    'From the moment I met him, I knew he was hiding something.',
    'The sky turned a strange shade of red, and I knew the end was near.',
    'It was a moment of triumph, and yet, a moment of despair.',
    'The air was thick with humidity, and the forest was teeming with life.',
    'Looking out over the ocean, I realized how vast the world truly was.',
    'The sound of sirens filled the air as chaos erupted all around me.',
    'The river flowed endlessly, carving its path through the mountains.',
    'I awoke in a cold, sterile room, unsure of how I had arrived there.',
    'As the clock struck twelve, I felt the weight of destiny pressing down on me.',
    'In the silence of the night, a distant howl sent chills down my spine.',
    'The rain tapped softly against the window as I sat alone in the dark.',
    'The old house creaked as if it were alive, each step making the floor groan beneath me.',
    'The moment our eyes met, I knew everything was about to change.',
    'A thick fog rolled in, covering the city in a blanket of mystery.',
    'I could feel the tension in the air as we waited for the inevitable.',
    'The sun was setting, casting long shadows across the barren landscape.',
    'As I reached the top of the mountain, the view took my breath away.',
    'The smell of smoke lingered in the air long after the fire had been put out.',
    'She smiled, but her eyes told a different story.',
    'I had never felt so alone, even though the room was full of people.',
    'The city lights flickered in the distance, a reminder of how far I had come.',
    'The cold wind bit at my face as I trudged through the snow.',
    'The sound of footsteps behind me made me quicken my pace.',
    'The clock was ticking, and time was running out.',
    'I knew I had to make a choice, but neither option felt right.',
    'The old man’s voice was shaky, but his words were filled with wisdom.',
    'The desert stretched out endlessly before me, with no sign of life in sight.',
    'I could hear the waves crashing in the distance, a soothing rhythm in the chaos.',
    'As the storm raged outside, I felt a strange calm settle over me.',
    'The letter arrived unexpectedly, throwing my world into disarray.',
    'The fire crackled softly, filling the cabin with warmth and light.',
    'The cold, damp air clung to my skin as I ventured deeper into the cave.',
    'The distant rumble of thunder warned of an approaching storm.',
    'The elevator doors opened, revealing a scene I could never have imagined.']
tokenizer = LlamaTokenizer.from_pretrained(llama7b_name)

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


def time_ssp(target_name, draft_name, K=4):
    draft_model = create_model(**models_params[draft_name])
    target_model = create_model(**models_params[target_name])
    nb_tokens = 0
    # Warmup
    input_ids = tokenizer(texts[0], return_tensors="pt").input_ids
    input_ids = torch.stack(
        [input_ids[0]] * batch_size).to(draft_model.device)
    generated_ids, accept_tokens, generated_tokens = ssp(target_model,
                        draft_model,
                        MAX_NEW_TOKENS,
                        input_ids, K=K)

    start_time = time.time()
    all_accept_tokens = 0
    all_generated_tokens = 0
    for text in texts[1:]:

        print("Completing text:", text)
        intermediate_time = time.time()
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = torch.stack(
            [input_ids[0]] * batch_size).to(draft_model.device)
        generated_ids, accept_tokens, generated_tokens = ssp(target_model,
                            draft_model,
                            MAX_NEW_TOKENS,
                            input_ids, K=K)
        all_accept_tokens += accept_tokens
        all_generated_tokens += generated_tokens
        nb_tokens += generated_ids.shape[1] - input_ids.shape[1]
        print("Completion: ", tokenizer.decode(
            generated_ids[0], skip_special_tokens=True))
        print("Time: {:.2f}s".format(time.time() - intermediate_time))
        print("========\n")
        print("Acceptance Rate: {:.2f}%".format((accept_tokens/generated_tokens)*100))
        print("========\n")
        
    ms_per_token = (time.time() - start_time)*1000 / nb_tokens
    accept_rate = (all_accept_tokens/all_generated_tokens)
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
    tokens_s, outputs, accept_rate = time_ssp(draft_name, target_name)
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
    # add argument to set log level
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
        gen_ids, ms_per_token, accept_rate = time_ssp(args.model, args.draft)
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

# Base functions for sampling and streaming

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# llama7b_name = 'baffo32/decapoda-research-llama-7b-hf'
opt_model_name = 'facebook/opt-1.3b'
tokenizer = AutoTokenizer.from_pretrained(opt_model_name)


def create_model(model_name, max_memory, device_map='balanced', offload_folder=None):
    kwargs = {
        "pretrained_model_name_or_path": model_name,
        "device_map": device_map,
        "max_memory": max_memory
    }
    
    # offload_folder가 지정된 경우에만 추가
    if offload_folder is not None:
        kwargs["offload_folder"] = offload_folder
        
    return AutoModelForCausalLM.from_pretrained(**kwargs)


def stream_token_if_required(input_ids, stream=False):
    if stream is True:
        output_string = tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True)
        previous_output_string = tokenizer.decode(
            input_ids[0][:-1],
            skip_special_tokens=True)
        sys.stdout.write(output_string[len(previous_output_string):])
        sys.stdout.flush()


TEMPERATURE = 0.0000000000000000001


def get_temperature_distribution(logits, temperature=TEMPERATURE):
    return torch.softmax(logits / temperature, dim=-1)


def sample_fn(logits, temperature=TEMPERATURE):
    probs = get_temperature_distribution(logits, temperature)
    # NaN과 inf 값을 처리
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    # 음수 값을 0으로 처리
    probs = torch.clamp(probs, min=0.0)
    # 확률 분포가 0이 되지 않도록 보정
    if probs.sum() == 0:
        probs = torch.ones_like(probs) / probs.size(-1)
    else:
        probs = probs / probs.sum()
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_model(model,
                 input_ids,
                 nb_tokens,
                 display=False,
                 temperature=TEMPERATURE):
    for _ in range(nb_tokens):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = sample_fn(next_token_logits, temperature)
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
        stream_token_if_required(input_ids, stream=display)
    return input_ids

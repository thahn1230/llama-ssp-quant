import torch
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoderLayer,
    OPTForCausalLM,
)
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant_2_bits import quantize_opt as quantize_opt_2
from smoothquant.fake_quant_4_bits import quantize_opt as quantize_opt_4
from smoothquant.fake_quant_5_bits import quantize_opt as quantize_opt_5
from smoothquant.fake_quant_6_bits import quantize_opt as quantize_opt_6
from smoothquant.fake_quant_7_bits import quantize_opt as quantize_opt_7
from smoothquant.fake_quant_8_bits import quantize_opt as quantize_opt_8


class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples["text"])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            
            # Check sequence length to avoid IndexError
            sequence_length = outputs.logits.size(1)
            if sequence_length < 2:
                # If the sequence length is less than 2, use the last token
                last_token_logits = outputs.logits[:, -1, :]
            else:
                # Otherwise, use the second-to-last token
                last_token_logits = outputs.logits[:, -2, :]
            
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc
        
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# dataset = dataset.select(range(1000))  # 데이터셋에서 처음 1000개를 선택
dataset = load_dataset("lambada", split="validation[:1000]")  # LAMBADA 데이터셋
evaluator = Evaluator(dataset, tokenizer, "cuda")

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b", torch_dtype=torch.float16, device_map="auto"
)

# opt_model = evaluator.evaluate(model)
# print(f"facebook/opt-1.3b model accuracy: {opt_model}")

act_scales = torch.load("act_scales/opt-6.7b-data_random44.pt")
smooth_lm(model, act_scales, 0.5)
model_smoothquant = quantize_opt_4(model)
# print(model_smoothquant_w8a8)

acc_smoothquant = evaluator.evaluate(model_smoothquant)
print(f"SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant}")
from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention
from transformers import AutoModelForCausalLM

# # 임의의 Qwen2 모델 불러오기
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
# attention_module = next(m for n, m in model.named_modules() if isinstance(m, Qwen2SdpaAttention))

# # Attention 모듈의 속성 확인
# print(attention_module)

from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

# Qwen2 모델 로드
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
mlp_module = next(m for n, m in model.named_modules() if isinstance(m, Qwen2MLP))

# Qwen2MLP 구조 확인
print(mlp_module)

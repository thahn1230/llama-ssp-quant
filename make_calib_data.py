import os
import json
from datasets import load_dataset

# 1. 원본 데이터셋 로드 및 일정 seed로 shuffle (동일 seed면 동일한 순서)
dataset = load_dataset("json", data_files="val.jsonl.zst", split="train")
dataset = dataset.shuffle(seed=42)
texts = dataset["text"][:1024]

# 2. acceptance_rates.json 파일 읽기 및 인덱스 추출
with open("acceptance_rates.json", "r", encoding="utf-8") as f:
    acceptance_data = json.load(f)

# acceptance_data는 이미 acceptance rate 순으로 정렬되어 있다고 가정
sorted_indexes = [item[0] for item in acceptance_data]

# 3. 추출한 인덱스 순서대로 텍스트 재정렬
sorted_texts = [texts[i] for i in sorted_indexes]

# 4. 512개씩 두 개의 데이터셋으로 분할
dataset1 = sorted_texts[:512]
dataset2 = sorted_texts[512:]

# 5. 각 데이터셋을 JSONL 파일로 저장 (하나의 파일 안에 각 데이터가 별도의 JSON 객체로 구분됨)
# 예시와 같이 "text"와 "meta" 정보를 포함시키도록 함.
def save_jsonl(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for text in dataset:
            record = {"text": text, "meta": {"pile_set_name": "OpenWebText2"}}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

save_jsonl(dataset1, "dataset1_44.jsonl")
save_jsonl(dataset2, "dataset2_44.jsonl")

print("각 데이터셋이 JSONL 파일로 저장되었습니다.")

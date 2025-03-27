import torch
import time

# CUDA가 사용 가능한지 확인
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(device)

# 큰 크기의 텐서 생성 (1000x1000)
tensor_a = torch.rand(1000, 1000, device=device)
tensor_b = torch.rand(1000, 1000, device=device)

# 행렬 곱셈 연산
start_time = time.time()
result = torch.matmul(tensor_a, tensor_b)
end_time = time.time()

print("연산 완료!")
print(f"소요 시간: {end_time - start_time}초")

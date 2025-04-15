import os
import argparse
import torch
import time
import json
import logging
import numpy as np
import random
import gc
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.multiprocessing as mp
import math

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 메모리 단편화 방지를 위한 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# llamassp 모듈에서 필요한 함수들 임포트
from llamassp import create_model, tokenizer, models_params, MAX_NEW_TOKENS
from lssp.ssp import ssp
from lssp.base import sample_model

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 불필요한 로깅 줄이기
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# 장치 설정 - 여러 GPU를 사용할 수 있도록 변경
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using CUDA with {torch.cuda.device_count()} available GPUs")
    # GPU 정보는 한 번만 간략하게 출력
    logger.info(f"Available GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
else:
    device = torch.device("cpu")
    logger.info("Using CPU - CUDA not available")

# 모델 파라미터 정의
models_params = {
    "13B": {
        "model_name": "facebook/opt-6.7b",
        "max_memory": {0: "22GB", 1: "22GB", 2: "22GB", 3: "22GB", 4: "22GB", 5: "22GB", 6: "22GB", 7: "22GB"},
        "device_map": "auto",
        "offload_folder": "offload_folder"
    },
    "7B": {
        "model_name": "facebook/opt-1.3b",
        "max_memory": {0: "22GB", 1: "22GB", 2: "22GB", 3: "22GB", 4: "22GB", 5: "22GB", 6: "22GB", 7: "22GB"},
        "device_map": "auto"
    },
    "3B": {
        "model_name": "facebook/opt-125m",
        "max_memory": {0: "22GB"},  # 단일 GPU에만 메모리 할당
        "device_map": 0  # 작은 모델은 단일 GPU에 할당
    }
}

def create_model_wrapper(**kwargs):
    """
    create_model을 래핑하는 함수
    """
    model_kwargs = kwargs.copy()
    
    logger.info(f"모델 로드 중: {model_kwargs.get('model_name', '알 수 없음')}")
    
    # create_model이 지원하지 않는 매개변수 제거
    for key in ['load_in_4bit', 'load_in_8bit', 'quantization_config', 'torch_dtype']:
        model_kwargs.pop(key, None)
    
    # 특수 처리: 3B 모델(opt-125m)의 경우 메모리가 적게 필요하므로 단일 GPU에 할당
    if model_kwargs.get('model_name') == 'facebook/opt-125m':
        logger.debug("작은 모델(opt-125m)은 단일 GPU에 배치합니다.")
        # GPU 0에 배치
        model_kwargs['device_map'] = 0
        
        # max_memory가 필수 파라미터이므로 항상 유지해야 함
        if 'max_memory' not in model_kwargs or not model_kwargs['max_memory']:
            model_kwargs['max_memory'] = {0: "22GB"}
    
    # 필수 파라미터 검사
    if 'max_memory' not in model_kwargs:
        logger.debug("max_memory 파라미터 누락됨. 기본값 설정: {0: '22GB'}")
        model_kwargs['max_memory'] = {0: "22GB"}
    
    # 모델 생성
    try:
        logger.debug(f"모델 로딩 파라미터: {model_kwargs}")
        model = create_model(**model_kwargs)
    except TypeError as e:
        logger.warning(f"모델 로딩 오류: {str(e)}")
        # 매개변수 오류 시 필요한 매개변수만 남기고 다시 시도
        minimal_kwargs = {
            'model_name': model_kwargs.get('model_name'),
            'max_memory': model_kwargs.get('max_memory', {0: "22GB"}),
            'device_map': model_kwargs.get('device_map', 0)
        }
        logger.debug("최소 매개변수로 재시도")
        model = create_model(**minimal_kwargs)
    
    # 디버깅 정보는 필요한 경우만 출력
    if logger.isEnabledFor(logging.DEBUG):
        if hasattr(model, 'hf_device_map'):
            logger.debug(f"모델 장치 맵: {model.hf_device_map}")
        else:
            try:
                logger.debug(f"모델 장치: {next(model.parameters()).device}")
            except Exception:
                logger.debug("모델 장치 정보를 가져올 수 없습니다.")
    
    return model

class ThresholdState:
    """임계값 상태를 표현하는 클래스"""
    def __init__(self, fallback, rollback):
        self.fallback = fallback
        self.rollback = rollback
    
    def __str__(self):
        return f"FB={self.fallback:.3f}, RB={self.rollback:.3f}"
    
    def as_dict(self):
        return {
            'fallback_threshold': self.fallback,
            'rollback_threshold': self.rollback
        }

class Experience:
    """경험 데이터를 저장하는 클래스"""
    def __init__(self, state, action, reward, next_state, metrics, old_log_prob):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.metrics = metrics
        self.old_log_prob = old_log_prob  # 이전 정책의 로그 확률 저장

class ExperienceReplay:
    """경험 재현 버퍼"""
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        self.episode_rewards = deque(maxlen=100)  # 최근 에피소드 보상 추적
    
    def add(self, experience):
        importance = abs(experience.reward)
        if len(self.episode_rewards) > 0:
            importance_factor = max(0.1, 1.0 + experience.reward / (np.mean(self.episode_rewards) + 1e-8))
            importance *= importance_factor
        self.buffer.append((experience, importance))
        self.episode_rewards.append(experience.reward)
    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        experiences, weights = zip(*self.buffer)
        weights = np.array(weights)
        weights = np.maximum(weights, 1e-8)
        probs = weights / (weights.sum() + 1e-8)
        if np.all(probs == 0):
            probs = np.ones_like(probs) / len(probs)
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs)
        return [experiences[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

def load_texts(dataset_name="lambada", max_samples=20):
    """테스트용 텍스트 샘플을 로드합니다."""
    from datasets import load_dataset
    
    if dataset_name == "lambada":
        dataset = load_dataset(dataset_name, split="test")
        texts = dataset["text"][:max_samples]
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [text for text in dataset["text"] if len(text.strip()) > 100][:max_samples]
    elif dataset_name == "c4":
        dataset = load_dataset("c4", "en", split="validation", streaming=True)
        texts = []
        count = 0
        for item in dataset:
            if len(item["text"].strip()) > 200:
                texts.append(item["text"])
                count += 1
                if count >= max_samples:
                    break
    else:
        raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
    
    return texts

def compute_reward(acceptance_rate, ms_per_token, speed_weight=0.7, acceptance_weight=0.3, 
                  baseline_ms=100.0, min_acceptance=0.3, episode=0, 
                  fallback_value=0.4, rollback_value=8.0, k_value=16):
    """개선된 보상 함수 - FB/RB 값에 따른 추가 보상/패널티"""
    if np.isnan(acceptance_rate) or np.isnan(ms_per_token):
        return 0.0  # 기본값 반환
    
    # 수락률이 최소 기준 미달시 큰 패널티
    if acceptance_rate < min_acceptance:
        penalty = -((min_acceptance - acceptance_rate) * 3.0) ** 2
        return penalty
    
    # 기본 보상 계산
    speed_score = baseline_ms / max(ms_per_token, 1.0)
    speed_score = np.tanh(speed_score - 1.0) + 1.0
    acceptance_score = acceptance_rate ** 0.4
    
    # FB/RB 값 보상 조정
    # 롤백이 K에 가까워질수록 패널티 (K의 50-70% 범위가 이상적)
    rb_ratio = rollback_value / k_value
    if rb_ratio > 0.8:  # K의 80% 초과시 패널티
        rb_penalty = (rb_ratio - 0.7) * 0.3
    elif rb_ratio < 0.3:  # K의 30% 미만시 패널티
        rb_penalty = (0.3 - rb_ratio) * 0.3
    else:
        rb_penalty = 0
    
    # 폴백 보상: 높을수록 속도 향상에 도움이 될 수 있지만,
    # 너무 높으면 생성 품질 저하 위험
    fb_bonus = 0
    if 0.4 <= fallback_value <= 0.7:  # 이상적인 FB 범위
        fb_bonus = 0.1
    
    # 최종 보상 계산
    base_reward = (speed_weight * speed_score) + (acceptance_weight * acceptance_score)
    adjusted_reward = base_reward - rb_penalty + fb_bonus
    
    return adjusted_reward

class PPOAgent:
    """Proximal Policy Optimization 알고리즘 구현"""
    
    def __init__(self, fallback_range=(0.05, 0.95), rollback_range=(1.0, 16.0),
                 lr=0.001, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, 
                 train_iterations=10, batch_size=64, k_value=16):
        self.fallback_range = fallback_range
        # 롤백 범위를 k_value로 제한
        self.rollback_range = (rollback_range[0], min(rollback_range[1], k_value))
        self.k_value = k_value  # K 값 저장 (드래프트 모델이 생성하는 최대 토큰 수)
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.train_iterations = train_iterations
        self.batch_size = batch_size
        
        # 각 파라미터별 영향력 조정 (FB가 더 큰 영향을 미치도록)
        self.fallback_delta_scale = 0.2  # FB 변화 스케일 증가 (0.1 → 0.2)
        self.rollback_delta_scale = 1.0  # RB 변화 스케일 유지
        
        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4)  # fallback_mean, fallback_std, rollback_mean, rollback_std
        ).to(device)  # 장치로 이동
        
        self.value_network = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ).to(device)  # 장치로 이동
        
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=lr)
        
        self.experience_buffer = ExperienceReplay(max_size=10000)
        
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'rewards': [],
            'advantages': []
        }
    
    def compute_gae(self, rewards, values, next_value, dones):
        """Generalized Advantage Estimation 계산"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)  # 장치로 이동
        returns = advantages + torch.tensor(values, dtype=torch.float32).to(device)  # 장치로 이동
        
        return advantages, returns
    
    def get_action(self, state, explore=True):
        """ 현재 정책에 따라 행동 선택 """
        with torch.no_grad():
            state_tensor = torch.FloatTensor([state.fallback, state.rollback]).unsqueeze(0).to(device)
            action_params = self.policy_network(state_tensor)
            
            # NaN 값 검사 및 처리
            if torch.isnan(action_params).any():
                logger.warning("NaN 값이 action_params에서 발생하여 기본값을 사용합니다.")
                # 기본값 설정
                fallback_mean = 0.0
                fallback_std = 0.2
                rollback_mean = 0.0
                rollback_std = 0.1
            else:
                # 정상 계산 진행
                fallback_mean = torch.tanh(action_params[0, 0]).item()
                fallback_std = torch.sigmoid(action_params[0, 1]).item() * 0.5
                rollback_mean = torch.tanh(action_params[0, 2]).item()
                rollback_std = torch.sigmoid(action_params[0, 3]).item() * 0.2
            
            # 최소 표준편차 보장
            fallback_std = max(fallback_std, 1e-6)
            rollback_std = max(rollback_std, 1e-6)
            
            # 분포 생성 및 샘플링
            try:
                # 탐색 모드일 때만 무작위 샘플링
                if explore:
                    # numpy 랜덤 샘플링으로 변경 (더 안정적)
                    fallback_action = np.random.normal(fallback_mean, fallback_std)
                    rollback_action = np.random.normal(rollback_mean, rollback_std)
                    
                    # 값 검증
                    if np.isnan(fallback_action) or np.isinf(fallback_action):
                        fallback_action = fallback_mean
                    if np.isnan(rollback_action) or np.isinf(rollback_action):
                        rollback_action = rollback_mean
                else:
                    # 탐색 안 할 때는 평균값 사용
                    fallback_action = fallback_mean
                    rollback_action = rollback_mean
                    
                # 범위 제한
                fallback_action = max(0.0, min(1.0, fallback_action))
                rollback_action = max(1.0, min(min(self.k_value, 16.0), rollback_action))
                
                # 스케일 적용 (기존 코드와 일관성 유지)
                if explore:
                    delta_fallback = fallback_action * self.fallback_delta_scale
                    delta_rollback = rollback_action * self.rollback_delta_scale
                else:
                    delta_fallback = fallback_mean * self.fallback_delta_scale
                    delta_rollback = rollback_mean * self.rollback_delta_scale
                
                # 현재 상태에서 델타 적용하여 새 상태 계산
                new_fallback = np.clip(state.fallback + delta_fallback, 
                                      self.fallback_range[0], self.fallback_range[1])
                
                # 롤백에는 제약 적용
                constrained_delta = delta_rollback * 0.5  # 롤백 변화 제한
                new_rollback = np.clip(state.rollback + constrained_delta, 
                                      self.rollback_range[0], self.rollback_range[1])
                
                # K를 초과하지 않도록 추가 체크
                new_rollback = min(new_rollback, self.k_value)
                
                # 액션은 새 상태 자체
                action = [new_fallback, new_rollback]
                
                # 로그 확률 계산 - numpy 사용
                # log_prob 계산 시 텐서 사용하지 않음
                log_prob = float(-(((fallback_action - fallback_mean) / fallback_std) ** 2) / 2 - np.log(fallback_std) - 0.5 * np.log(2 * np.pi))
                log_prob += float(-(((rollback_action - rollback_mean) / rollback_std) ** 2) / 2 - np.log(rollback_std) - 0.5 * np.log(2 * np.pi))
                
            except Exception as e:
                logger.warning(f"분포 생성 또는 샘플링 중 오류 발생: {e}")
                # 기본값 직접 계산
                new_fallback = state.fallback
                new_rollback = state.rollback
                action = [new_fallback, new_rollback]
                log_prob = 0.0
            
            return action, log_prob
    
    def update(self):
        """개선된 PPO 업데이트"""
        if len(self.experience_buffer) < self.batch_size // 2:  # 버퍼 크기 요구사항 완화
            return
        
        recent_experiences = self.experience_buffer.sample(self.batch_size // 2)
        old_experiences = self.experience_buffer.sample(self.batch_size // 2)
        batch = recent_experiences + old_experiences
        
        states = torch.FloatTensor([[exp.state.fallback, exp.state.rollback] for exp in batch]).to(device)  # 장치로 이동
        actions = torch.FloatTensor([exp.action for exp in batch]).to(device)  # 장치로 이동
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(device)  # 장치로 이동
        next_states = torch.FloatTensor([[exp.next_state.fallback, exp.next_state.rollback] 
                                        for exp in batch]).to(device)  # 장치로 이동
        dones = torch.FloatTensor([0.0] * len(batch)).to(device)  # 장치로 이동
        
        with torch.no_grad():
            values = self.value_network(states).squeeze()
            next_value = self.value_network(next_states[-1].unsqueeze(0)).squeeze()
        
        # NaN 값 검사 및 처리
        if torch.isnan(values).any() or torch.isnan(next_value).any():
            logger.warning("Value network에서 NaN 값이 발생하여 업데이트를 건너뜁니다.")
            return
        
        advantages, returns = self.compute_gae(rewards, values, next_value, dones)
        
        if torch.isnan(advantages).any() or torch.isnan(returns).any():
            logger.warning("GAE 계산 중 NaN 값이 발생하여 업데이트를 건너뜁니다.")
            return
        
        # 수치 안정성을 위한 스케일링
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        if torch.isnan(advantages).any():
            logger.warning("NaN detected in advantages. Skipping update.")
            return
        
        if len(self.training_history['rewards']) > 100:
            recent_rewards = self.training_history['rewards'][-100:]
            if np.mean(recent_rewards) < np.mean(self.training_history['rewards'][:-100]):
                self.lr *= 0.98
                for param_group in self.policy_optimizer.param_groups:
                    param_group['lr'] = self.lr
                for param_group in self.value_optimizer.param_groups:
                    param_group['lr'] = self.lr
        
        for _ in range(self.train_iterations):
            self.policy_optimizer.zero_grad()
            action_params = self.policy_network(states)
            
            # NaN 검사
            if torch.isnan(action_params).any():
                logger.warning("Policy network에서 NaN 값이 발생하여 업데이트를 건너뜁니다.")
                return
            
            # 표준편차가 0이 되지 않도록 하여 안정성 보장
            fallback_means = torch.tanh(action_params[:, 0])
            fallback_stds = torch.sigmoid(action_params[:, 1]) * 0.5
            rollback_means = torch.tanh(action_params[:, 2])
            rollback_stds = torch.sigmoid(action_params[:, 3]) * 0.2
            
            # 최소 표준편차 적용
            fallback_stds = torch.clamp(fallback_stds, min=1e-6)
            rollback_stds = torch.clamp(rollback_stds, min=1e-6)
            
            if torch.isnan(fallback_means).any() or torch.isnan(fallback_stds).any() or \
               torch.isnan(rollback_means).any() or torch.isnan(rollback_stds).any():
                logger.warning("NaN detected in action parameters. Skipping update.")
                return
            
            # 기존 코드와 동일
            fallback_dist = torch.distributions.Normal(fallback_means, fallback_stds)
            rollback_dist = torch.distributions.Normal(rollback_means, rollback_stds)
            
            fallback_log_probs = fallback_dist.log_prob(actions[:, 0])
            rollback_log_probs = rollback_dist.log_prob(actions[:, 1])
            
            new_log_probs = fallback_log_probs + rollback_log_probs
            
            # nan 값이 있는지 확인
            if torch.isnan(new_log_probs).any():
                logger.warning("NaN log probabilities detected. Skipping update.")
                return
            
            old_log_probs = torch.FloatTensor([exp.old_log_prob for exp in batch]).to(device)  # 장치로 이동
            
            # 무한대 값이나 NaN이 있는지 확인
            if torch.isinf(old_log_probs).any() or torch.isnan(old_log_probs).any():
                logger.warning("Invalid values in old log probabilities. Resetting to zero.")
                old_log_probs = torch.zeros_like(old_log_probs)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 비율이 너무 크거나 NaN인 경우 클리핑
            if torch.isnan(ratio).any() or (ratio > 100.0).any():
                logger.warning("매우 큰 비율이나 NaN 값이 발견되어 조정합니다.")
                ratio = torch.clamp(ratio, 0.01, 100.0)  # 극단적인 값 방지
            
            clip_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            
            policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
            
            # NaN이나 무한대 값이 있는지 확인
            if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                logger.warning("Invalid policy loss detected. Skipping backpropagation.")
                continue
            
            policy_loss.backward()
            # 기울기 클리핑 강화
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            values = self.value_network(states).squeeze()
            value_loss = torch.nn.MSELoss()(values, returns)
            
            # NaN이나 무한대 값이 있는지 확인
            if torch.isnan(value_loss) or torch.isinf(value_loss):
                logger.warning("Invalid value loss detected. Skipping backpropagation.")
                continue
            
            value_loss.backward()
            # 기울기 클리핑 강화
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
            self.value_optimizer.step()
            
            self.training_history['policy_loss'].append(policy_loss.item())
            self.training_history['value_loss'].append(value_loss.item())
            self.training_history['advantages'].append(advantages.mean().item())
        
        self.training_history['rewards'].extend(rewards.tolist())
        
        if len(self.experience_buffer) > self.batch_size * 2:
            self.experience_buffer.buffer = deque(list(self.experience_buffer.buffer)[-self.batch_size:])
        
        # 학습 후 모델 파라미터 검사
        with torch.no_grad():
            for name, param in self.policy_network.named_parameters():
                if torch.isnan(param).any():
                    logger.warning(f"NaN 값이 정책 네트워크의 {name} 파라미터에서 발견되었습니다. 재설정합니다.")
                    param.data.copy_(torch.randn_like(param) * 0.01)  # 작은 랜덤 값으로 초기화
    
    def save_model(self, filepath):
        """모델 저장"""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'training_history': self.training_history
        }, filepath)
    
    def load_model(self, filepath):
        """모델 로드"""
        checkpoint = torch.load(filepath)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.training_history = checkpoint['training_history']
    
    def plot_training_history(self, save_path=None):
        """훈련 이력 시각화"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.training_history['policy_loss'])
        plt.title('Policy Loss')
        plt.xlabel('Updates')
        
        plt.subplot(1, 3, 2)
        plt.plot(self.training_history['value_loss'])
        plt.title('Value Loss')
        plt.xlabel('Updates')
        
        plt.subplot(1, 3, 3)
        plt.plot(self.training_history['rewards'])
        plt.title('Rewards')
        plt.xlabel('Episodes')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def reward_function(self, state, k_value=16):
        """임계값 설정의 보상 계산 함수"""
        # 각 범위 내에서의 적절함 평가
        
        # 1. 폴백 임계값 평가
        if 0.3 <= state.fallback <= 0.7:
            fb_score = 0.3  # 적절한 범위
        elif state.fallback < 0.2 or state.fallback > 0.8:
            fb_score = -0.2  # 극단적인 범위는 불리
        else:
            fb_score = 0.1  # 중간 범위
        
        # 2. 롤백 임계값 평가
        rb_ratio = state.rollback / k_value
        
        if 0.3 <= rb_ratio <= 0.7:
            rb_score = 0.3  # K의 30-70% 범위가 이상적
        elif rb_ratio < 0.1 or rb_ratio > 0.9:
            rb_score = -0.3  # 매우 극단적인 범위는 매우 불리
        else:
            rb_score = 0.0  # 중간 범위
        
        # 보상 점수의 합
        reward = fb_score + rb_score
        
        # K값을 넘지 않는 보너스
        if state.rollback <= k_value:
            reward += 0.1
        
        return reward

    def collect_trajectory(self, env, max_steps=24):
        """환경과 상호작용하여 궤적 수집"""
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        
        # 초기 상태 설정
        state = ThresholdState(fallback=0.4, rollback=min(self.k_value/2, 8.0))
        
        for _ in range(max_steps):
            # 현재 정책으로 행동 선택
            action_values, log_prob = self.get_action(state)
            
            # 보상 계산 - 환경 시뮬레이션
            reward = self.reward_function(ThresholdState(action_values[0], action_values[1]), self.k_value)
            
            # 메모리에 저장
            states.append([state.fallback, state.rollback])
            actions.append(action_values)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(False)
            
            # 다음 스텝
            state = ThresholdState(action_values[0], action_values[1])
        
        # 마지막 상태에 대한 처리
        dones[-1] = True
        
        # 텐서 변환
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.FloatTensor(actions).to(device)
        log_probs_tensor = torch.FloatTensor(log_probs).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        dones_tensor = torch.FloatTensor(dones).to(device)
        
        return states_tensor, actions_tensor, log_probs_tensor, rewards_tensor, dones_tensor

def process_batch(texts, target_model, draft_model, state, K, max_new_tokens):
    """배치 단위로 텍스트 처리"""
    results = []
    batch_size = 1  # 배치 크기를 최소화하여 메모리 사용량 감소
    
    # 모델 관련 디바이스 정보 확인 (디버그 로그 제거)
    target_embed_device = None
    draft_embed_device = None
    
    # 임베딩 레이어 장치 찾기 (입력 처리를 위해 중요)
    if hasattr(target_model, 'model') and hasattr(target_model.model, 'decoder') and hasattr(target_model.model.decoder, 'embed_tokens'):
        target_embed_device = target_model.model.decoder.embed_tokens.weight.device
    elif hasattr(target_model, 'hf_device_map'):
        # 임베딩 레이어의 장치 찾기
        if 'model.decoder.embed_tokens' in target_model.hf_device_map:
            target_embed_device = torch.device(target_model.hf_device_map['model.decoder.embed_tokens'])
    
    if hasattr(draft_model, 'model') and hasattr(draft_model.model, 'decoder') and hasattr(draft_model.model.decoder, 'embed_tokens'):
        draft_embed_device = draft_model.model.decoder.embed_tokens.weight.device
    elif hasattr(draft_model, 'hf_device_map'):
        # 임베딩 레이어의 장치 찾기
        if 'model.decoder.embed_tokens' in draft_model.hf_device_map:
            draft_embed_device = torch.device(draft_model.hf_device_map['model.decoder.embed_tokens'])
    
    # 임베딩 장치를 찾지 못한 경우 기본값 사용
    if target_embed_device is None:
        target_embed_device = torch.device('cuda:0')
    
    if draft_embed_device is None:
        # 타겟 모델과 같은 장치 사용
        draft_embed_device = target_embed_device
    
    # 장치 정보 간단히 로깅
    logger.debug(f"Target: {target_embed_device}, Draft: {draft_embed_device}")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_results = []
        
        for text in batch_texts:
            try:
                # 토크나이저 결과를 타겟 모델의 임베딩 레이어와 동일한 장치로 이동
                input_tensor = tokenizer(
                    text, 
                    return_tensors="pt", 
                    max_length=512,  
                    truncation=True  
                )
                
                # 장치 이동
                input_ids = input_tensor.input_ids.to(target_embed_device)
                
                start_time = time.time()
                
                # SSP 실행
                try:
                    generated_ids, accept_tokens, generated_tokens = ssp(
                        target_model,
                        draft_model,
                        max_new_tokens,
                        input_ids,
                        K=K,
                        fallback_threshold=state.fallback,
                        rollback_threshold=state.rollback
                    )
                except RuntimeError as e:
                    if "Expected all tensors to be on the same device" in str(e):
                        # 오류 발생 시 draft_model을 target_model의 임베딩 장치로 이동 시도
                        logger.warning(f"장치 불일치 오류, 드래프트 모델 이동: {target_embed_device}")
                        if hasattr(draft_model, 'to'):
                            draft_model_device_before = next(draft_model.parameters()).device
                            draft_model = draft_model.to(target_embed_device)
                        
                        # 다시 시도
                        generated_ids, accept_tokens, generated_tokens = ssp(
                            target_model,
                            draft_model,
                            max_new_tokens,
                            input_ids,
                            K=K,
                            fallback_threshold=state.fallback,
                            rollback_threshold=state.rollback
                        )
                
                elapsed = time.time() - start_time
                
                num_new_tokens = generated_ids.shape[1] - input_ids.shape[1]
                
                if num_new_tokens > 0:
                    ms_per_token = (elapsed * 1000) / num_new_tokens
                else:
                    ms_per_token = 1000.0
                    
                acceptance_rate = accept_tokens / max(generated_tokens, 1)
                
                batch_results.append({
                    'ms_per_token': ms_per_token,
                    'acceptance_rate': acceptance_rate
                })
                
                # 메모리 해제
                del input_ids, generated_ids
                # 명시적 가비지 컬렉션
                if i % 5 == 0:  # 5개 배치마다 가비지 컬렉션 실행
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"텍스트 처리 중 오류 발생: {str(e)}")
                # 스택 트레이스는 디버그 로그에만 출력
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception("오류 상세 정보:")
                batch_results.append({
                    'ms_per_token': 1000.0,
                    'acceptance_rate': 0.0
                })
        
        results.extend(batch_results)
    
    # 처리 완료 후 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def evaluate_thresholds_parallel(target_model, draft_model, state, texts, K=16, max_new_tokens=64, episode=0):
    """병렬 처리로 임계값 평가 - 메모리 효율성을 위해 최대 새 토큰 수 감소"""
    # 메모리 사용량 감소를 위해 병렬 프로세스 수 제한
    num_processes = min(2, len(texts))
    chunk_size = len(texts) // num_processes
    
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i in range(num_processes):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_processes - 1 else len(texts)
            chunk_texts = texts[start_idx:end_idx]
            
            future = executor.submit(
                process_batch,
                chunk_texts,
                target_model,
                draft_model,
                state,
                K,
                max_new_tokens
            )
            futures.append(future)
        
        all_results = []
        for future in as_completed(futures):
            all_results.extend(future.result())
    
    avg_ms_per_token = sum(r['ms_per_token'] for r in all_results) / len(all_results)
    avg_acceptance_rate = sum(r['acceptance_rate'] for r in all_results) / len(all_results)
    
    # 수정된 계산: FB/RB 값도 보상에 반영
    reward = compute_reward(
        avg_acceptance_rate, 
        avg_ms_per_token,
        speed_weight=0.7, 
        acceptance_weight=0.3, 
        episode=episode,
        fallback_value=state.fallback,
        rollback_value=state.rollback,
        k_value=K
    )
    
    metrics = {
        'ms_per_token': avg_ms_per_token,
        'acceptance_rate': avg_acceptance_rate,
        'reward': reward
    }
    
    # 간결한 로깅 출력
    logger.info(f"평가: FB={state.fallback:.2f}, RB={state.rollback:.1f} → 수락={avg_acceptance_rate:.2f}, 속도={avg_ms_per_token:.1f}ms, 보상={reward:.2f}")
    
    return reward, metrics

def train_model_pair(target_name, draft_name, agent, texts, episodes=20, K=16, max_new_tokens=64):
    """특정 모델 쌍에 대한 에이전트 훈련 - 메모리 효율성을 위해 max_new_tokens 감소"""
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    # 여러 GPU를 사용할 수 있도록 설정
    if torch.cuda.is_available():
        logger.debug(f"Using all available GPUs: {torch.cuda.device_count()}")
    else:
        logger.debug("CUDA not available, using CPU")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"모델 로드 중: {target_name} (타겟), {draft_name} (드래프트) - 시도 {attempt + 1}/{max_retries}")
            
            # 원래 파라미터 사용
            target_params = models_params[target_name].copy()
            draft_params = models_params[draft_name].copy()
                
            target_model = create_model_wrapper(**target_params)
            draft_model = create_model_wrapper(**draft_params)
            
            # 모델 장치 정보 확인
            logger.info(f"모델 로드 완료")
            
            # 임베딩 레이어 장치 확인 (입력 텐서와 일치해야 함)
            target_embed_device = None
            draft_embed_device = None
            
            if hasattr(target_model, 'model') and hasattr(target_model.model, 'decoder') and hasattr(target_model.model.decoder, 'embed_tokens'):
                target_embed_device = target_model.model.decoder.embed_tokens.weight.device
                logger.debug(f"Target model embedding device: {target_embed_device}")
            elif hasattr(target_model, 'hf_device_map') and 'model.decoder.embed_tokens' in target_model.hf_device_map:
                target_embed_device = torch.device(target_model.hf_device_map['model.decoder.embed_tokens'])
                logger.debug(f"Target model embedding device from map: {target_embed_device}")
            
            if hasattr(draft_model, 'model') and hasattr(draft_model.model, 'decoder') and hasattr(draft_model.model.decoder, 'embed_tokens'):
                draft_embed_device = draft_model.model.decoder.embed_tokens.weight.device
                logger.debug(f"Draft model embedding device: {draft_embed_device}")
            elif hasattr(draft_model, 'hf_device_map') and 'model.decoder.embed_tokens' in draft_model.hf_device_map:
                draft_embed_device = torch.device(draft_model.hf_device_map['model.decoder.embed_tokens'])
                logger.debug(f"Draft model embedding device from map: {draft_embed_device}")
            
            # 작은 모델인 경우 전체 모델을 같은 장치로 이동
            if draft_name == "3B" and target_embed_device is not None:
                # opt-125m 모델은 작아서 전체를 같은 장치로 이동 가능
                logger.debug(f"작은 드래프트 모델(3B)을 타겟 임베딩과 같은 장치로 이동: {target_embed_device}")
                if draft_embed_device != target_embed_device:
                    draft_model = draft_model.to(target_embed_device)
                    logger.debug(f"Draft model moved to {target_embed_device}")
            
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"모델 로드 실패: {str(e)}. {retry_delay}초 후 재시도...")
                time.sleep(retry_delay)
            else:
                logger.error(f"모델 로드 최종 실패: {str(e)}")
                raise
    
    results = []
    
    # 초기 임계값 설정: 폴백은 중간 정도, 롤백은 K의 절반 정도로 설정
    state = ThresholdState(fallback=0.4, rollback=min(K/2, 8.0))
    
    exploration_rate = 1.0
    exploration_decay = 0.95
    
    for episode in range(episodes):
        logger.info(f"에피소드 {episode+1}/{episodes} 시작 - 모델: {target_name}/{draft_name}")
        
        if random.random() < exploration_rate:
            # 랜덤 탐색 시에도 RB는 K 이하로 유지
            next_state = ThresholdState(
                fallback=random.uniform(0.3, 0.5),
                rollback=random.uniform(1.0, min(K, 8.0))  # K를 초과하지 않도록 제한
            )
            # 랜덤 탐색 시에는 직접 old_log_prob 계산
            old_log_prob = 0.0  # 랜덤 탐색이므로 로그 확률은 0으로 설정
        else:
            # 새로운 get_action 사용
            action_values, old_log_prob = agent.get_action(state)
            
            # 새로운 상태 생성
            new_fallback = max(0.0, min(1.0, action_values[0]))
            new_rollback = max(0.0, min(min(K, 16.0), action_values[1]))
            next_state = ThresholdState(new_fallback, new_rollback)
        
        exploration_rate *= exploration_decay
        
        for eval_attempt in range(max_retries):
            try:
                reward, metrics = evaluate_thresholds_parallel(
                    target_model, draft_model, next_state, texts, K, max_new_tokens, episode
                )
                # 수정된 보상 계산: FB/RB 값 전달
                reward = compute_reward(
                    metrics['acceptance_rate'], 
                    metrics['ms_per_token'],
                    speed_weight=0.7,
                    acceptance_weight=0.3,
                    episode=episode,
                    fallback_value=next_state.fallback,
                    rollback_value=next_state.rollback,
                    k_value=K
                )
                break
            except Exception as e:
                if eval_attempt < max_retries - 1:
                    logger.warning(f"평가 실패: {str(e)}. {retry_delay}초 후 재시도...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"평가 최종 실패: {str(e)}")
                    raise
        
        agent.experience_buffer.add(
            Experience(state, (next_state.fallback - state.fallback, next_state.rollback - state.rollback), 
                    reward, next_state, metrics, old_log_prob)
        )
        
        agent.update()
        
        result = {
            'episode': episode + 1,
            'target': target_name,
            'draft': draft_name,
            'fallback_threshold': next_state.fallback,
            'rollback_threshold': next_state.rollback,
            'ms_per_token': metrics['ms_per_token'],
            'acceptance_rate': metrics['acceptance_rate'],
            'reward': reward,
            'exploration_rate': exploration_rate
        }
        results.append(result)
        
        state = next_state
        
        # 에피소드 완료 후 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # GPU 메모리 사용량 로깅 (디버그 레벨로 변경)
        if torch.cuda.is_available() and logger.isEnabledFor(logging.DEBUG):
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                reserved_mem = torch.cuda.memory_reserved(i) / 1024**3
                allocated_mem = torch.cuda.memory_allocated(i) / 1024**3
                gpu_info.append(f"GPU {i}: {allocated_mem:.1f}/{reserved_mem:.1f}/{total_mem:.1f}GB")
            logger.debug(f"GPU Memory: {', '.join(gpu_info)}")
    
    return results

def meta_learning(datasets, model_pairs, args):
    """다양한 모델/데이터셋에 대한 메타러닝 수행"""
    
    # K 값 전달하여 PPOAgent 초기화
    agent = PPOAgent(k_value=args.k)
    
    all_results = []
    best_thresholds = {}
    
    for dataset_name, texts in datasets.items():
        logger.info(f"데이터셋 {dataset_name}에 대한 훈련 시작")
        
        for target_name, draft_name in model_pairs:
            logger.info(f"\n{'='*20} 모델 쌍 {target_name}/{draft_name} 훈련 {'='*20}")
            
            pair_results = train_model_pair(
                target_name, draft_name, agent, texts, 
                episodes=args.episodes_per_pair,
                K=args.k,
                max_new_tokens=args.max_tokens
            )
            
            best_result = max(pair_results, key=lambda x: x['reward'])
            best_thresholds[f"{target_name}_{draft_name}_{dataset_name}"] = {
                'fallback': best_result['fallback_threshold'],
                'rollback': best_result['rollback_threshold'],
                'acceptance_rate': best_result['acceptance_rate'],
                'ms_per_token': best_result['ms_per_token'],
                'reward': best_result['reward']
            }
            
            # 훈련 완료 후 최적 결과 출력
            logger.info(f"최적 임계값: FB={best_result['fallback_threshold']:.2f}, RB={best_result['rollback_threshold']:.1f}")
            logger.info(f"수락률={best_result['acceptance_rate']:.2f}, 속도={best_result['ms_per_token']:.1f}ms, 보상={best_result['reward']:.2f}")
            
            for result in pair_results:
                result['dataset'] = dataset_name
                all_results.append(result)
    
    agent.save_model(args.model_output)
    agent.plot_training_history(args.model_output.replace('.pt', '_history.png'))
    
    return all_results, best_thresholds, agent

def cross_validation(agent, datasets, model_pairs, args):
    """학습된 에이전트 크로스 검증"""
    
    validation_results = []
    
    # 여러 GPU를 사용할 수 있도록 설정
    if torch.cuda.is_available():
        logger.debug(f"Validation using all available GPUs: {torch.cuda.device_count()}")
    else:
        logger.debug("CUDA not available for validation, using CPU")
    
    for dataset_name, texts in datasets.items():
        for target_name, draft_name in model_pairs:
            logger.info(f"검증: {target_name}/{draft_name} on {dataset_name}")
            
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    # 원래 파라미터 사용
                    target_params = models_params[target_name].copy()
                    draft_params = models_params[draft_name].copy()
                        
                    target_model = create_model_wrapper(**target_params)
                    draft_model = create_model_wrapper(**draft_params)
                    
                    # 모델 장치 정보는 디버그 레벨에서만 출력
                    logger.info(f"검증 모델 로드 완료")
                    
                    # 임베딩 레이어 장치 확인 (입력 텐서와 일치해야 함)
                    target_embed_device = None
                    draft_embed_device = None
                    
                    if hasattr(target_model, 'model') and hasattr(target_model.model, 'decoder') and hasattr(target_model.model.decoder, 'embed_tokens'):
                        target_embed_device = target_model.model.decoder.embed_tokens.weight.device
                        logger.debug(f"Target model embedding device: {target_embed_device}")
                    elif hasattr(target_model, 'hf_device_map') and 'model.decoder.embed_tokens' in target_model.hf_device_map:
                        target_embed_device = torch.device(target_model.hf_device_map['model.decoder.embed_tokens'])
                        logger.debug(f"Target model embedding device from map: {target_embed_device}")
                    
                    if hasattr(draft_model, 'model') and hasattr(draft_model.model, 'decoder') and hasattr(draft_model.model.decoder, 'embed_tokens'):
                        draft_embed_device = draft_model.model.decoder.embed_tokens.weight.device
                        logger.debug(f"Draft model embedding device: {draft_embed_device}")
                    elif hasattr(draft_model, 'hf_device_map') and 'model.decoder.embed_tokens' in draft_model.hf_device_map:
                        draft_embed_device = torch.device(draft_model.hf_device_map['model.decoder.embed_tokens'])
                        logger.debug(f"Draft model embedding device from map: {draft_embed_device}")
                    
                    # 작은 모델인 경우 전체 모델을 같은 장치로 이동
                    if draft_name == "3B" and target_embed_device is not None:
                        # opt-125m 모델은 작아서 전체를 같은 장치로 이동 가능
                        logger.debug(f"작은 드래프트 모델(3B)을 타겟 임베딩과 같은 장치로 이동: {target_embed_device}")
                        if draft_embed_device != target_embed_device:
                            draft_model = draft_model.to(target_embed_device)
                            logger.debug(f"Draft model moved to {target_embed_device}")
                            
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"모델 로드 실패: {str(e)}. {retry_delay}초 후 재시도...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"모델 로드 최종 실패: {str(e)}")
                        raise
            
            # 초기 상태를 K 값에 맞게 조정
            init_state = ThresholdState(fallback=0.4, rollback=min(args.k/2, 8.0))
            
            # 새로운 get_action 메소드 사용
            action_values, _ = agent.get_action(init_state)
            
            # 새로운 상태 생성
            new_fallback = max(0.0, min(1.0, action_values[0]))
            new_rollback = max(0.0, min(min(args.k, 16.0), action_values[1]))
            recommended_state = ThresholdState(new_fallback, new_rollback)
            
            # K를 초과하지 않도록 제한
            if recommended_state.rollback > args.k:
                logger.warning(f"롤백 값이 K({args.k})를 초과하여 {args.k}로 제한합니다.")
                recommended_state = ThresholdState(
                    recommended_state.fallback, 
                    args.k
                )
            
            for eval_attempt in range(max_retries):
                try:
                    reward, metrics = evaluate_thresholds_parallel(
                        target_model, draft_model, recommended_state, texts,
                        K=args.k, max_new_tokens=args.max_tokens
                    )
                    break
                except Exception as e:
                    if eval_attempt < max_retries - 1:
                        logger.warning(f"평가 실패: {str(e)}. {retry_delay}초 후 재시도...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"평가 최종 실패: {str(e)}")
                        raise
            
            result = {
                'target': target_name,
                'draft': draft_name,
                'dataset': dataset_name,
                'fallback_threshold': recommended_state.fallback,
                'rollback_threshold': recommended_state.rollback,
                'ms_per_token': metrics['ms_per_token'],
                'acceptance_rate': metrics['acceptance_rate'],
                'reward': reward
            }
            validation_results.append(result)
            
            # 검증 완료 후 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # GPU 메모리 사용량 로깅 (디버그 레벨로 변경)
            if torch.cuda.is_available() and logger.isEnabledFor(logging.DEBUG):
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    reserved_mem = torch.cuda.memory_reserved(i) / 1024**3
                    allocated_mem = torch.cuda.memory_allocated(i) / 1024**3
                    gpu_info.append(f"GPU {i}: {allocated_mem:.1f}/{reserved_mem:.1f}/{total_mem:.1f}GB")
                logger.debug(f"GPU Memory: {', '.join(gpu_info)}")
    
    return validation_results

def main():
    parser = argparse.ArgumentParser(description="Advanced RL for threshold optimization")
    parser.add_argument('--targets', type=str, nargs='+', required=True, 
                        help='Target model names')
    parser.add_argument('--drafts', type=str, nargs='+', required=True, 
                        help='Draft model names')
    parser.add_argument('--datasets', type=str, nargs='+', default=["lambada", "wikitext"],
                        help='Datasets to use for training')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples per dataset')
    parser.add_argument('--k', type=int, default=16,
                        help='Number of tokens for draft model to generate at once')
    parser.add_argument('--max-tokens', type=int, default=64,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--episodes-per-pair', type=int, default=5,
                        help='Number of episodes per model pair')
    parser.add_argument('--output', type=str, default="meta_learning_results.json",
                        help='Output file for results')
    parser.add_argument('--model-output', type=str, default="threshold_agent.pt",
                        help='Output file for trained model')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce logging output')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize thresholds')
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
        # 추가로 다른 라이브러리의 로깅 레벨도 조정
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("accelerate").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)
    
    # 데이터셋 로드 정보
    datasets = {}
    for dataset_name in args.datasets:
        logger.info(f"{dataset_name} 데이터셋 로드 중...")
        texts = load_texts(dataset_name=dataset_name, max_samples=args.samples)
        logger.info(f"{len(texts)} 텍스트 샘플 로드 완료")
        datasets[dataset_name] = texts
    
    # 모델 쌍 구성
    model_pairs = list(product(args.targets, args.drafts))
    logger.info(f"총 {len(model_pairs)}개 모델 쌍에 대해 훈련: {model_pairs}")
    
    # 메타러닝 수행
    print(f"\n{'='*30} 메타러닝 시작 {'='*30}")
    all_results, best_thresholds, agent = meta_learning(datasets, model_pairs, args)
    
    # 크로스 검증
    print(f"\n{'='*30} 크로스 검증 시작 {'='*30}")
    validation_results = cross_validation(agent, datasets, model_pairs, args)
    
    # 결과 저장
    final_results = {
        'training': all_results,
        'best_thresholds': best_thresholds,
        'validation': validation_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # 결과 요약 출력
    print("\n" + "=" * 50)
    print("메타러닝 결과 요약:")
    print("-" * 50)
    
    # 최적 임계값 요약
    print("모델별 최적 임계값:")
    for model_key, thresholds in best_thresholds.items():
        print(f"* {model_key}:")
        print(f"  - Fallback: {thresholds['fallback']:.3f}, Rollback: {thresholds['rollback']:.3f}")
        print(f"  - 수락률: {thresholds['acceptance_rate']:.3f}, 속도: {thresholds['ms_per_token']:.2f}ms/token")
        print(f"  - 보상: {thresholds['reward']:.3f}")
    
    # 검증 결과 요약
    print("\n크로스 검증 결과:")
    for result in validation_results:
        print(f"* {result['target']}/{result['draft']} on {result['dataset']}:")
        print(f"  추천: FB={result['fallback_threshold']:.3f}, RB={result['rollback_threshold']:.3f}, "
              f"수락률={result['acceptance_rate']:.3f}, 속도={result['ms_per_token']:.2f}ms/token")
    
    # 일반화된 추천값
    print("\n일반화된 추천 임계값 (모든 모델에 적용 가능):")
    avg_fallback = sum(r['fallback_threshold'] for r in validation_results) / len(validation_results)
    avg_rollback = sum(r['rollback_threshold'] for r in validation_results) / len(validation_results)
    print(f"Fallback: {avg_fallback:.3f}, Rollback: {avg_rollback:.3f}")
    print("=" * 50)

    # 최적의 임계값 추천
    if args.optimize:
        print("\n최적 임계값 계산 중...")
        for k in sorted(k_values):
            args.k = k
            agent = PPOAgent(args)
            agent.load_model(f"models/ppo_agent_k{k}.pt")
            
            # 초기 상태를 K 값에 맞게 조정
            init_state = ThresholdState(fallback=0.4, rollback=min(args.k/2, 8.0))
            action_values, _ = agent.get_action(init_state)
            
            # 새로운 상태 생성
            new_fallback = max(0.0, min(1.0, action_values[0]))
            new_rollback = max(0.0, min(min(args.k, 16.0), action_values[1]))
            recommended_state = ThresholdState(new_fallback, new_rollback)
            
            # K를 초과하지 않도록 제한
            recommended_state.rollback = min(recommended_state.rollback, k)
            
            print(f"K={k}: 폴백 임계값={recommended_state.fallback:.4f}, 롤백 임계값={recommended_state.rollback:.4f}")

if __name__ == "__main__":
    main() 
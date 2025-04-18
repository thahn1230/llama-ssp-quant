import os
import argparse
import torch
import time
import json
import logging
import numpy as np
import random
import gc
import multiprocessing
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.multiprocessing as mp
import math
from matplotlib.gridspec import GridSpec
from concurrent.futures import ProcessPoolExecutor
import sys
from datetime import datetime
import glob

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # INFO 레벨로 변경 (DEBUG에서 상향)
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
        logging.FileHandler('ssp_debug.log')  # 파일 출력
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 메모리 단편화 방지를 위한 설정 - 멀티프로세싱 환경에서 문제가 발생하므로 수정
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 기본 설정으로 변경
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

# 토크나이저 임포트
from transformers import AutoTokenizer

# llamassp 모듈에서 필요한 함수들 임포트
from llamassp import create_model, tokenizer as llamassp_tokenizer, models_params, MAX_NEW_TOKENS

# 토크나이저가 없거나 오류가 발생한 경우 직접 초기화
try:
    tokenizer = llamassp_tokenizer
    # 간단한 테스트
    test_text = "Hello, world!"
    test_tokens = tokenizer(test_text, return_tensors="pt")
    logger.info(f"임포트된 토크나이저 사용: {tokenizer.__class__.__name__}")
except (ImportError, AttributeError, NameError, TypeError) as e:
    logger.warning(f"임포트된 토크나이저 오류: {str(e)}, AutoTokenizer 사용")
    # 기본 OPT 토크나이저 사용
    default_model_name = "facebook/opt-6.7b"
    tokenizer = AutoTokenizer.from_pretrained(default_model_name)
    logger.info(f"AutoTokenizer 초기화 완료: {tokenizer.__class__.__name__}")

# 시간 통계 수집을 지원하는 ssp 함수 사용 (없으면 기본 ssp 사용)
try:
    from lssp.ssp_modified import ssp
    HAS_MODIFIED_SSP = True
    logger.info("Using modified SSP with time statistics support")
except ImportError:
    from lssp.ssp import ssp
    HAS_MODIFIED_SSP = False
    logger.warning("Using standard SSP without time statistics support")
    
from lssp.base import sample_model

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
    def __init__(self, fallback, rollback, time_stats=None):
        self.fallback = fallback
        self.rollback = rollback
        # 시간 통계 정보 저장
        self.time_stats = time_stats if time_stats else {}
    
    def __str__(self):
        return f"FB={self.fallback:.3f}, RB={self.rollback:.3f}"
    
    def as_dict(self):
        result = {
            'fallback_threshold': self.fallback,
            'rollback_threshold': self.rollback
        }
        # 시간 통계 추가
        if self.time_stats:
            result['time_stats'] = self.time_stats
        return result

class Experience:
    """강화학습 경험 데이터 - 레이턴시 최적화 중심으로 수정"""
    
    def __init__(self, state, action, reward, next_state, metrics, old_log_prob):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.metrics = metrics
        self.old_log_prob = old_log_prob
    
    def get_time_stats(self):
        """시간 통계 정보 반환 (없으면 None)"""
        if self.metrics and 'time_stats' in self.metrics:
            return self.metrics['time_stats']
        return None
    
    def get_latency(self):
        """레이턴시 정보 반환 (없으면 None)"""
        if self.metrics and 'ms_per_token' in self.metrics:
            return self.metrics['ms_per_token']
        return None
    
    def get_acceptance_rate(self):
        """수락률 정보 반환 (없으면 None)"""
        if self.metrics and 'acceptance_rate' in self.metrics:
            return self.metrics['acceptance_rate']
        return None

class ExperienceReplay:
    """강화학습 경험 버퍼링 - 레이턴시 최적화를 위한 우선순위 샘플링 추가"""
    
    def __init__(self, max_size=10000):
        """초기화"""
        self.buffer = []
        self.max_size = max_size
    
    def add(self, experience):
        """경험 추가"""
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # 가장 오래된 경험 제거
    
    def sample(self, batch_size):
        """경험 샘플링 - 레이턴시 기반 우선순위"""
        if len(self.buffer) < batch_size:
            # 버퍼 크기가 충분하지 않은 경우 모든 샘플 반환
            return self.buffer.copy()
        
        # 레이턴시(ms_per_token) 기반 샘플링 가중치 계산
        # 낮은 레이턴시(좋은 성능)를 가진 경험에 더 높은 가중치 부여
        latencies = [exp.get_latency() for exp in self.buffer]
        if not latencies or all(np.isnan(lat) for lat in latencies):
            # 유효한 레이턴시 정보가 없으면 균등 샘플링
            return random.sample(self.buffer, batch_size)
            
        max_latency = max(lat for lat in latencies if not np.isnan(lat))
        if max_latency == 0:
            # 모든 레이턴시가 0이면 균등 샘플링
            return random.sample(self.buffer, batch_size)
            
        # 가중치 계산: 레이턴시가 낮을수록 더 높은 확률
        weights = []
        for lat in latencies:
            if np.isnan(lat) or lat <= 0:
                # 유효하지 않은 레이턴시는 중간 가중치 부여
                weights.append(0.5)
            else:
                # 레이턴시가 낮을수록 높은 가중치 (1에 가까움)
                weight = 1.0 - (lat / max_latency)
                weights.append(max(0.1, weight))  # 최소 0.1 가중치 보장
        
        # 가중치 기반 샘플링
        try:
            sampled_indices = random.choices(range(len(self.buffer)), weights=weights, k=batch_size)
            return [self.buffer[i] for i in sampled_indices]
        except ValueError:
            # 가중치 기반 샘플링에 실패한 경우 균등 샘플링으로 폴백
            logger.warning("가중치 기반 샘플링 실패, 균등 샘플링으로 대체합니다.")
            return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """버퍼 크기 반환"""
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

def compute_reward(acceptance_rate, ms_per_token, time_stats=None, 
                  min_acceptance=0.0, baseline_ms=100.0, episode=0, 
                  fallback_value=0.4, rollback_value=8.0, k_value=16):
    """레이턴시와 수락률의 균형을 고려한 보상 함수 - 강화학습 개선"""
    
    # 입력값 유효성 검사
    if ms_per_token is None or np.isnan(ms_per_token):
        logger.warning(f"ms_per_token이 유효하지 않음: {ms_per_token}")
        return -10.0  # 기본 음수 보상
    
    # 매우 높은 레이턴시 페널티 
    if ms_per_token > 500.0:  # 과도하게 높은 레이턴시
        logger.warning(f"과도한 레이턴시: {ms_per_token:.2f}ms")
        return -10.0  # 즉시 큰 페널티
    
    # ==== 레이턴시 점수 계산 (낮을수록 높은 점수) ====
    # 기준값: ~100ms 정도가 좋은 성능이라고 가정
    # 로그 스케일 사용: 작은 개선도 보상하고, 큰 악화는 크게 페널티
    latency_score = 5.0 - np.log10(max(10.0, ms_per_token)) * 3.0
    # 범위 제한: -5.0 ~ 5.0
    latency_score = np.clip(latency_score, -5.0, 5.0)
    
    # ==== 수락률 점수 계산 (적절한 범위가 최적) ====
    acceptance_score = 0.0
    if acceptance_rate is not None and not np.isnan(acceptance_rate):
        # 최적 수락률 범위: 0.3 ~ 0.8 (너무 낮거나 높지 않은 범위)
        if 0.3 <= acceptance_rate <= 0.8:
            # 최적 범위 내에서는 수락률이 높을수록 더 높은 보상
            acceptance_score = 3.0 * (acceptance_rate - 0.3) / 0.5  # 0 ~ 3.0 점수
        elif acceptance_rate < 0.3:
            # 너무 낮은 수락률은 페널티
            acceptance_score = -5.0 * (1.0 - acceptance_rate / 0.3)  # -5.0 ~ 0 점수
        else:  # > 0.8
            # 너무 높은 수락률은 약한 페널티 (드래프트가 타겟과 너무 비슷하거나 임계값이 너무 높음)
            acceptance_score = 3.0 - 2.0 * (acceptance_rate - 0.8) / 0.2  # 3.0 ~ 1.0 점수
    
    # ==== 임계값 보너스 계산 ====
    # 경험적으로 좋은 임계값 범위에 보너스 부여
    threshold_bonus = 0.0
    
    # Fallback 임계값 범위에 따른 보너스 (0.1~0.25가 최적 범위)
    if 0.1 <= fallback_value <= 0.25:
        threshold_bonus += 1.0
        logger.info(f"Fallback 임계값 최적 범위 보너스: +1.0")
    elif fallback_value < 0.1:
        # 너무 낮은 fallback은 불필요한 검증 증가
        penalty = -2.0 * (0.1 - fallback_value) / 0.1  # 최대 -2.0
        threshold_bonus += penalty
        logger.info(f"Fallback 임계값 너무 낮음 페널티: {penalty:.2f}")
    elif fallback_value > 0.35:
        # 너무 높은 fallback은 너무 많은 롤백 유발
        penalty = -3.0 * (fallback_value - 0.35) / 0.15  # 최대 -3.0
        threshold_bonus += penalty
        logger.info(f"Fallback 임계값 너무 높음 페널티: {penalty:.2f}")
    
    # Rollback 임계값 범위에 따른 보너스 (k_value의 15~30%가 최적)
    optimal_lower = 0.15 * k_value
    optimal_upper = 0.30 * k_value
    
    if optimal_lower <= rollback_value <= optimal_upper:
        threshold_bonus += 1.0
        logger.info(f"Rollback 임계값 최적 범위 보너스: +1.0")
    elif rollback_value < optimal_lower and rollback_value >= 1.0:
        # 너무 낮은 rollback은 너무 짧은 수락으로 처리량 감소
        penalty = -1.0 * (optimal_lower - rollback_value) / optimal_lower  # 최대 -1.0
        threshold_bonus += penalty
        logger.info(f"Rollback 임계값 너무 낮음 페널티: {penalty:.2f}")
    elif rollback_value > optimal_upper:
        # 너무 높은 rollback은 너무 많은 토큰 낭비
        penalty = -2.0 * (rollback_value - optimal_upper) / (k_value - optimal_upper)  # 최대 -2.0
        threshold_bonus += penalty
        logger.info(f"Rollback 임계값 너무 높음 페널티: {penalty:.2f}")
    
    # ==== 가중치 계산 (에피소드에 따라 조정) ====
    # 학습 초기: 균형있는 탐색 유도
    # 학습 후기: 레이턴시 최적화에 집중
    latency_weight = 0.5 + min(0.3, episode * 0.01)   # 0.5 → 0.8로 증가
    acceptance_weight = 1.0 - latency_weight          # 0.5 → 0.2로 감소
    
    # ==== 최종 보상 계산 ====
    weighted_score = (latency_weight * latency_score) + (acceptance_weight * acceptance_score)
    final_reward = weighted_score + threshold_bonus
    
    # ==== 로깅 ====
    logger.info(f"레이턴시 점수: {latency_score:.2f} (ms_per_token: {ms_per_token:.2f}ms)")
    logger.info(f"수락률 점수: {acceptance_score:.2f} (acceptance_rate: {acceptance_rate:.2f})")
    logger.info(f"임계값 보너스: {threshold_bonus:.2f} (FB={fallback_value:.3f}, RB={rollback_value:.2f})")
    logger.info(f"가중치: 레이턴시={latency_weight:.2f}, 수락률={acceptance_weight:.2f}")
    logger.info(f"최종 보상: {weighted_score:.2f}(가중치 적용) + {threshold_bonus:.2f}(임계값) = {final_reward:.2f}")
    
    return float(final_reward)

class PPOAgent:
    """Proximal Policy Optimization 알고리즘 구현"""
    
    def __init__(self, fallback_range=(0.05, 0.5), rollback_range=(0.5, 10.0),
                 lr=0.002, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, 
                 train_iterations=8, batch_size=64, k_value=16):
        """PPO 에이전트 초기화 - 레이턴시 최적화 및 안정성 개선"""
        # 임계값 범위 설정
        self.fallback_range = fallback_range  # 더 작은 범위로 조정
        # 롤백 범위를 k_value로 제한
        self.rollback_range = (rollback_range[0], min(rollback_range[1], k_value))
        self.k_value = k_value  # K 값 저장
        
        # PPO 하이퍼파라미터
        self.lr = lr  # 약간 더 큰 학습률
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.train_iterations = train_iterations  # 반복 횟수 감소 (과적합 방지)
        self.batch_size = batch_size
        
        # 각 파라미터별 영향력 조정
        self.fallback_delta_scale = 0.1  # FB 변화 스케일 감소 (더 작은 스텝)
        self.rollback_delta_scale = 0.5  # RB 변화 스케일 감소 (더 작은 스텝)
        
        # 정책 네트워크 - 레이어 사이즈 증가
        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4)  # fallback_mean, fallback_std, rollback_mean, rollback_std
        ).to(device)
        
        # 가치 네트워크 - 레이어 사이즈 증가
        self.value_network = torch.nn.Sequential(
            torch.nn.Linear(2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ).to(device)
        
        # 옵티마이저 - 더 작은 weight_decay 추가 (정규화)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), 
                                               lr=lr, weight_decay=1e-5)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), 
                                              lr=lr, weight_decay=1e-5)
        
        # 경험 버퍼
        self.experience_buffer = ExperienceReplay(max_size=10000)
        
        # 학습 이력 기록
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'rewards': [],
            'advantages': [],
            'latency': [],
            'acceptance': []
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
    
    def collect_trajectory(self, env, max_steps=24):
        """환경에서 trajectory 수집"""
        initial_state = env.reset()
        trajectory = []
        
        if max_steps < 1:
            max_steps = 24
        
        try:
            # 상태, 액션, 보상 기록
            for step in range(max_steps):
                try:
                    # 상태 유효성 검사 및 수정
                    if not hasattr(initial_state, 'fallback') or not hasattr(initial_state, 'rollback'):
                        break
                    
                    if math.isnan(initial_state.fallback) or math.isnan(initial_state.rollback):
                        if math.isnan(initial_state.fallback):
                            initial_state.fallback = 0.3
                        if math.isnan(initial_state.rollback):
                            initial_state.rollback = 4.0
                
                    # 액션 선택
                    action, log_prob = self.get_action(initial_state, explore=True)
                    
                    # 액션 유효성 검사 및 수정
                    if math.isnan(action[0]) or math.isnan(action[1]) or math.isnan(log_prob):
                        if math.isnan(action[0]):
                            action[0] = initial_state.fallback
                        if math.isnan(action[1]):
                            action[1] = initial_state.rollback
                        if math.isnan(log_prob):
                            log_prob = 0.0
                
                    # 환경에서 다음 상태와 보상 얻기
                    next_state, reward, metrics = env.step(action)
                    
                    # 보상과 다음 상태 유효성 검사 및 수정
                    if math.isnan(reward):
                        reward = 0.0
                    
                    if not hasattr(next_state, 'fallback') or not hasattr(next_state, 'rollback'):
                        break
                    
                    if math.isnan(next_state.fallback) or math.isnan(next_state.rollback):
                        if math.isnan(next_state.fallback):
                            next_state.fallback = action[0]
                        if math.isnan(next_state.rollback):
                            next_state.rollback = action[1]
                
                    # Experience 객체 생성 및 추가
                    exp = Experience(initial_state, action, reward, next_state, metrics, log_prob)
                    trajectory.append(exp)
                    
                    # 다음 스텝을 위해 상태 업데이트
                    initial_state = next_state
                except:
                    break
            
            return trajectory
            
        except:
            return []
    
    def get_action(self, state, explore=True):
        """현재 정책에 따라 행동 선택 - 개선된 탐색 전략 구현"""
        try:
            with torch.no_grad():
                # 상태 텐서 변환
                state_tensor = torch.FloatTensor([state.fallback, state.rollback]).unsqueeze(0).to(device)
                
                # 액션 분포 파라미터 계산
                action_params = self.policy_network(state_tensor)
                
                # 파라미터 변환 - 더 넓은 범위 및 미세 조정 지원
                # fallback은 0.05~0.4 범위
                fallback_mean = torch.sigmoid(action_params[0, 0]).item() * 0.35 + 0.05
                # 학습 단계에 따라 표준편차 조정 - 초기에는 넓게, 후기에는 좁게
                if len(self.training_history['rewards']) < 50:  # 초기 학습
                    fallback_std = torch.sigmoid(action_params[0, 1]).item() * 0.08 + 0.03  # 0.03~0.11
                else:  # 후기 학습
                    fallback_std = torch.sigmoid(action_params[0, 1]).item() * 0.04 + 0.01  # 0.01~0.05
                
                # rollback은 k_value의 10%~40% 범위로 조정
                rb_ratio_mean = torch.sigmoid(action_params[0, 2]).item() * 0.3 + 0.1  # 0.1~0.4
                rollback_mean = rb_ratio_mean * self.k_value
                
                # 학습 단계에 따라 표준편차 조정
                if len(self.training_history['rewards']) < 50:  # 초기 학습
                    rollback_std = torch.sigmoid(action_params[0, 3]).item() * 1.5 + 0.5  # 0.5~2.0
                else:  # 후기 학습
                    rollback_std = torch.sigmoid(action_params[0, 3]).item() * 0.8 + 0.2  # 0.2~1.0
                
                # explore 모드인 경우 가끔 더 극단적인 값도 시도 (탐색 강화)
                if explore and random.random() < 0.1:  # 10% 확률로 극단 탐색
                    # 임의로 네 가지 영역 중 하나 선택 (0: 낮은 FB, 1: 높은 FB, 2: 낮은 RB, 3: 높은 RB)
                    extreme_area = random.randint(0, 3)
                    
                    if extreme_area == 0:  # 매우 낮은 fallback 탐색
                        fallback_action = random.uniform(self.fallback_range[0], 0.1)
                        rollback_action = rollback_mean
                    elif extreme_area == 1:  # 매우 높은 fallback 탐색 
                        fallback_action = random.uniform(0.35, self.fallback_range[1])
                        rollback_action = rollback_mean
                    elif extreme_area == 2:  # 매우 낮은 rollback 탐색
                        fallback_action = fallback_mean
                        rollback_action = random.uniform(1.0, self.k_value * 0.15)
                    else:  # 매우 높은 rollback 탐색
                        fallback_action = fallback_mean
                        rollback_action = random.uniform(self.k_value * 0.35, self.rollback_range[1])
                    
                    logger.info(f"극단적 탐색 선택: 영역={extreme_area}, FB={fallback_action:.3f}, RB={rollback_action:.2f}")
                    
                    # 로그 확률은 임의로 낮게 설정 (이 액션의 영향력 감소)
                    log_prob = -5.0  
                # 일반적인 샘플링 또는 평균값 사용    
                elif explore:
                    # 비대칭 정규분포 샘플링 (인접한 두 정규분포 중 적절한 쪽 선택)
                    # Fallback 값 샘플링 - 좋은 범위(0.1-0.25)에 더 높은 확률
                    if random.random() < 0.7:  # 70% 확률로 좋은 범위 중심으로 샘플링
                        good_fb_mean = (0.1 + 0.25) / 2
                        fallback_action = np.random.normal(good_fb_mean, 0.05)
                    else:  # 30% 확률로 정책에서 제안한 값 기준 샘플링
                        fallback_action = np.random.normal(fallback_mean, fallback_std)
                    
                    # Rollback 값 샘플링 - K의 15-30%에 더 높은 확률
                    if random.random() < 0.7:  # 70% 확률로 좋은 범위 중심으로 샘플링 
                        good_rb_mean = (0.15 + 0.30) / 2 * self.k_value
                        rollback_action = np.random.normal(good_rb_mean, self.k_value * 0.05)
                    else:  # 30% 확률로 정책에서 제안한 값 기준 샘플링
                        rollback_action = np.random.normal(rollback_mean, rollback_std)
                    
                    # 로그 확률 계산 (단순화)
                    log_prob_fallback = -0.5 * ((fallback_action - fallback_mean) / fallback_std) ** 2
                    log_prob_rollback = -0.5 * ((rollback_action - rollback_mean) / rollback_std) ** 2
                    log_prob = float(log_prob_fallback + log_prob_rollback - np.log(fallback_std) - np.log(rollback_std)) - 2.0
                else:
                    # 탐색 안함 - 현재 최선의 추정치 사용
                    fallback_action = fallback_mean
                    rollback_action = rollback_mean
                    log_prob = 0.0  # 평균값 사용 시 로그 확률은 무의미
                
                # 범위 제약
                fallback_action = np.clip(fallback_action, self.fallback_range[0], self.fallback_range[1])
                rollback_action = np.clip(rollback_action, self.rollback_range[0], self.rollback_range[1])
                
                # k_value 고려하여 롤백 값 조정
                rollback_action = min(rollback_action, self.k_value)
                rollback_action = max(rollback_action, 1.0)  # 최소 1
                
                # 값 확인 (NaN, inf 등)
                if np.isnan(log_prob) or np.isinf(log_prob):
                    log_prob = 0.0
                
                # 결과 반환
                return [fallback_action, rollback_action], log_prob
        
        except Exception as e:
            logger.error(f"액션 생성 중 오류: {str(e)}")
            # 안전한 기본값 반환
            fallback = np.clip(0.2, self.fallback_range[0], self.fallback_range[1])  # 안전한 기본값 0.2
            rollback = np.clip(self.k_value * 0.25, self.rollback_range[0], min(self.rollback_range[1], self.k_value))  # 25% 기본값
            return [fallback, rollback], 0.0
    
    def update(self):
        """수집된 경험을 기반으로 정책과 가치 네트워크 업데이트 - 학습 안정성 개선"""
        if len(self.experience_buffer) < self.batch_size:
            return  # 충분한 경험이 없으면 학습 건너뜀
        
        try:
            # 학습 데이터 수집 및 전처리
            batch = self.experience_buffer.sample(self.batch_size)
            
            # 성공적인 경험 비율 확인 및 로깅
            success_rewards = [exp.reward for exp in batch if exp.reward > 0]
            success_ratio = len(success_rewards) / len(batch) if batch else 0
            logger.info(f"배치 내 성공 경험 비율: {success_ratio:.2f} ({len(success_rewards)}/{len(batch)})")
            
            # 학습 가능 여부 판단 - 최소한의 성공 경험 필요
            if success_ratio < 0.1 and len(self.training_history['rewards']) > 10:
                logger.warning("성공 경험이 너무 적어 학습을 건너뜁니다")
                return
            
            # 보상 값 범위 확인하여 정규화 필요 여부 결정
            rewards_array = np.array([exp.reward for exp in batch])
            reward_min, reward_max = rewards_array.min(), rewards_array.max()
            reward_range = reward_max - reward_min
            
            # 보상 범위가 너무 크면 경고 로깅
            if reward_range > 20.0:
                logger.warning(f"보상 범위가 매우 큽니다: {reward_min:.2f} ~ {reward_max:.2f}")
            
            # 경험 가중치 계산: 보상 기반 (높은 보상의 경험에 더 높은 가중치)
            if reward_range > 1e-5:  # 0으로 나누기 방지
                # 보상 정규화 및 가중치 계산
                normalized_rewards = (rewards_array - reward_min) / reward_range
                # 지수 함수로 높은 보상에 더 높은 가중치
                reward_weights = np.exp(2.0 * normalized_rewards)
            else:
                # 보상 범위가 너무 작으면 균등 가중치
                reward_weights = np.ones_like(rewards_array)
            
            # 레이턴시 기반 가중치 계산
            latencies = np.array([exp.get_latency() for exp in batch])
            latency_min, latency_max = latencies.min(), latencies.max()
            latency_range = latency_max - latency_min
            
            if latency_range > 1e-5:
                # 레이턴시 정규화 (낮을수록 높은 가중치)
                normalized_latencies = 1.0 - (latencies - latency_min) / latency_range
                latency_weights = np.exp(normalized_latencies) 
            else:
                latency_weights = np.ones_like(latencies)
            
            # 가중치 결합 (보상과 레이턴시를 모두 고려)
            combined_weights = 0.7 * reward_weights + 0.3 * latency_weights
            # 가중치 정규화 (합이 1)
            sample_weights = combined_weights / combined_weights.sum()
            
            # 텐서 변환
            states = torch.FloatTensor([[exp.state.fallback, exp.state.rollback] for exp in batch]).to(device)
            actions = torch.FloatTensor([exp.action for exp in batch]).to(device)
            rewards = torch.FloatTensor([exp.reward for exp in batch]).to(device)
            next_states = torch.FloatTensor([[exp.next_state.fallback, exp.next_state.rollback] for exp in batch]).to(device)
            old_log_probs = torch.FloatTensor([exp.old_log_prob for exp in batch]).to(device)
            weights_tensor = torch.FloatTensor(sample_weights).to(device)
            
            # 가치 네트워크 예측
            values = self.value_network(states).squeeze()
            next_values = self.value_network(next_states).squeeze().detach()
            
            # GAE 계산
            dones = torch.zeros(self.batch_size).to(device)  # 모든 상태 전환은 에피소드가 끝나지 않음을 가정
            advantages, returns = self.compute_gae(rewards.cpu().numpy(), values.cpu().detach().numpy(), 
                                                next_values.cpu().numpy(), dones.cpu().numpy())
            
            # 어드밴티지 정규화 (학습 안정성 개선)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 학습 지표 저장
            self.training_history['rewards'].append(rewards.mean().item())
            self.training_history['advantages'].append(advantages.mean().item())
            
            # 평균 레이턴시 및 수락률 기록
            avg_latency = np.mean([exp.get_latency() for exp in batch])
            avg_acceptance = np.mean([exp.get_acceptance_rate() for exp in batch])
            self.training_history['latency'].append(avg_latency)
            self.training_history['acceptance'].append(avg_acceptance)
            
            # 학습률 감소 (후반부 학습 안정화)
            if len(self.training_history['rewards']) > 100:
                current_lr = max(self.lr * 0.5, 0.0005)  # 학습률 감소 (최소 0.0005)
                for param_group in self.policy_optimizer.param_groups:
                    param_group['lr'] = current_lr
                for param_group in self.value_optimizer.param_groups:
                    param_group['lr'] = current_lr
                    
                if len(self.training_history['rewards']) % 50 == 0:
                    logger.info(f"학습률 조정: {current_lr:.6f}")
            
            policy_losses = []
            value_losses = []
            entropy_terms = []
            
            # 여러 에포크 동안 학습
            for _ in range(self.train_iterations):
                # 현재 정책으로 액션 분포 파라미터 계산
                action_means, action_stds = self._get_policy_parameters(states)
                
                # 새 정책의 로그 확률 계산
                dist = torch.distributions.Normal(action_means, action_stds)
                new_log_probs = dist.log_prob(actions).sum(dim=1)
                
                # 엔트로피 계산 (탐색 촉진)
                entropy = dist.entropy().sum(dim=1)
                
                # 새 정책과 이전 정책의 비율
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # 클리핑된 서라운드 목적 함수
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                
                # 가중치를 적용한 정책 손실 함수
                policy_loss = -torch.min(surr1, surr2) * weights_tensor
                
                # 엔트로피 보너스 추가 (초기에 더 크게, 후반에 작게)
                entropy_coef = max(0.01, 0.1 - 0.001 * len(self.training_history['rewards']))
                entropy_bonus = -entropy_coef * entropy
                
                # 정책 손실에 엔트로피 보너스 추가
                policy_loss = policy_loss + entropy_bonus * weights_tensor
                policy_loss = policy_loss.mean()
                
                # 가치 함수 손실
                value_pred = self.value_network(states).squeeze()
                value_loss = ((value_pred - returns) ** 2) * weights_tensor
                value_loss = value_loss.mean()
                
                # 정책 네트워크 업데이트
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                # 그래디언트 클리핑 추가
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
                self.policy_optimizer.step()
                
                # 가치 네트워크 업데이트
                self.value_optimizer.zero_grad()
                value_loss.backward()
                # 그래디언트 클리핑 추가
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
                self.value_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_terms.append(entropy.mean().item())
            
            # 학습 결과 저장
            self.training_history['policy_loss'].append(np.mean(policy_losses))
            self.training_history['value_loss'].append(np.mean(value_losses))
            
            # 학습 결과 로깅
            entropy_avg = np.mean(entropy_terms)
            logger.info(f"업데이트 완료: 정책 손실={np.mean(policy_losses):.4f}, 가치 손실={np.mean(value_losses):.4f}, 엔트로피={entropy_avg:.4f}")
            logger.info(f"평균 보상={rewards.mean().item():.4f}, 보상 범위: {reward_min:.2f}~{reward_max:.2f}")
            logger.info(f"평균 레이턴시={avg_latency:.2f}ms, 평균 수락률={avg_acceptance*100:.1f}%")
            # 가중치 정보 로깅
            logger.info(f"샘플 가중치: 최소={sample_weights.min():.3f}, 최대={sample_weights.max():.3f}")
            
            return np.mean(policy_losses), np.mean(value_losses)
            
        except Exception as e:
            logger.error(f"업데이트 중 오류 발생: {str(e)}")
            # 스택 트레이스 출력
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
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
        """학습 이력 시각화 - 레이턴시와 수락률 추가"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            plt.figure(figsize=(15, 10))
            gs = GridSpec(3, 2)
            
            # 보상 그래프
            ax1 = plt.subplot(gs[0, 0])
            rewards = self.training_history['rewards']
            if rewards:
                # 이동 평균 계산 (부드러운 곡선을 위해)
                window_size = min(10, len(rewards))
                if window_size > 0:
                    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    ax1.plot(smoothed_rewards, label='Smoothed Reward')
                ax1.plot(rewards, label='Reward', alpha=0.3)
                ax1.set_title('Rewards over Training')
                ax1.set_xlabel('Updates')
                ax1.set_ylabel('Reward')
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.7)
            
            # 손실 함수 그래프
            ax2 = plt.subplot(gs[0, 1])
            if self.training_history['policy_loss'] and self.training_history['value_loss']:
                ax2.plot(self.training_history['policy_loss'], label='Policy Loss')
                ax2.plot(self.training_history['value_loss'], label='Value Loss')
                ax2.set_title('Loss Functions')
                ax2.set_xlabel('Updates')
                ax2.set_ylabel('Loss')
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 레이턴시 그래프 - 새 추가
            ax3 = plt.subplot(gs[1, 0])
            latencies = self.training_history.get('latency', [])
            if latencies:
                # 이동 평균 계산
                window_size = min(5, len(latencies))
                if window_size > 0:
                    smoothed_latencies = np.convolve(latencies, np.ones(window_size)/window_size, mode='valid')
                    ax3.plot(smoothed_latencies, label='Smoothed Latency')
                ax3.plot(latencies, label='Latency', alpha=0.3)
                ax3.set_title('Latency over Training')
                ax3.set_xlabel('Updates')
                ax3.set_ylabel('ms per token')
                ax3.legend()
                ax3.grid(True, linestyle='--', alpha=0.7)
            
            # 수락률 그래프 - 새 추가
            ax4 = plt.subplot(gs[1, 1])
            acceptance = self.training_history.get('acceptance', [])
            if acceptance:
                # 이동 평균 계산
                window_size = min(5, len(acceptance))
                if window_size > 0:
                    smoothed_acceptance = np.convolve(acceptance, np.ones(window_size)/window_size, mode='valid')
                    ax4.plot(smoothed_acceptance, label='Smoothed Acceptance Rate')
                ax4.plot(acceptance, label='Acceptance Rate', alpha=0.3)
                ax4.set_title('Acceptance Rate over Training')
                ax4.set_xlabel('Updates')
                ax4.set_ylabel('Acceptance Rate')
                ax4.legend()
                ax4.grid(True, linestyle='--', alpha=0.7)
            
            # 레이턴시 vs 수락률 산점도 - 새 추가
            ax5 = plt.subplot(gs[2, :])
            if latencies and acceptance:
                ax5.scatter(acceptance, latencies, alpha=0.6, c='blue')
                ax5.set_title('Latency vs Acceptance Rate')
                ax5.set_xlabel('Acceptance Rate')
                ax5.set_ylabel('Latency (ms per token)')
                ax5.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
                logger.info(f"학습 이력 그래프가 {save_path}에 저장되었습니다.")
            else:
                plt.show()
        
        except Exception as e:
            logger.error(f"학습 이력 시각화 중 오류: {str(e)}")

    def reward_function(self, state, k_value=16, time_stats=None):
        """임계값 설정에 따른 보상 - 단순화된 버전"""
        # 기본 보상 (이 함수는 실제 레이턴시 정보가 없어 보수적으로 설정)
        base_reward = -5.0  # 보수적인 기본값
        
        # 임계값 기반 보너스
        threshold_bonus = 0.0
        
        # 경험적으로 좋은 임계값 범위에 보너스 부여
        if 0.1 <= state.fallback <= 0.3:
            threshold_bonus += 1.0
        
        # 롤백은 K의 10-30% 범위가 좋은 경우가 많음
        rb_ratio = state.rollback / k_value
        if 0.1 <= rb_ratio <= 0.3:
            threshold_bonus += 1.0
        
        # 최종 보상
        return base_reward + threshold_bonus

    def analyze_time_stats(self, time_stats):
        """시간 통계 분석 및 임계값 추천"""
        if not time_stats:
            return {
                'efficiency': 0.0,
                'bottleneck': 'unknown',
                'suggestions': ['시간 통계가 없어 분석할 수 없습니다.']
            }
        
        total_time = sum(time_stats.values())
        if total_time == 0:
            return {
                'efficiency': 0.0,
                'bottleneck': 'unknown',
                'suggestions': ['시간 측정이 없습니다.']
            }
        
        # 각 단계별 비율 계산
        time_ratios = {k: v/total_time for k, v in time_stats.items()}
        
        # 비효율성 발견 및 추천사항
        suggestions = []
        bottleneck = max(time_ratios.items(), key=lambda x: x[1])[0]
        
        # 효율성 점수 (0-1)
        ideal_ratios = {
            'draft_generation': 0.25,     # 25%
            'target_verification': 0.4,   # 40%
            'fallback_handling': 0.05,    # 5%
            'rollback_handling': 0.1,     # 10%
            'token_acceptance': 0.15,     # 15%
            'overhead': 0.05              # 5%
        }
        
        # 이상적인 비율과의 차이 계산
        ratio_diffs = {}
        for k in ideal_ratios.keys():
            actual = time_ratios.get(k, 0)
            ideal = ideal_ratios.get(k, 0)
            ratio_diffs[k] = abs(actual - ideal)
        
        # 효율성 점수 (1에 가까울수록 이상적)
        efficiency = 1.0 - sum(ratio_diffs.values()) / len(ratio_diffs)
        
        # 문제 영역 식별 및 추천
        if time_ratios.get('draft_generation', 0) > 0.4:
            suggestions.append("드래프트 모델 생성에 너무 많은 시간이 소모됩니다. 더 작은 드래프트 모델을 사용하거나, 폴백 임계값을 높이세요.")
        
        if time_ratios.get('fallback_handling', 0) > 0.2:
            suggestions.append("폴백 처리에 너무 많은 시간이 소모됩니다. 폴백 임계값을 높여 폴백 빈도를 줄이세요.")
        
        if time_ratios.get('rollback_handling', 0) > 0.25:
            suggestions.append("롤백 처리에 너무 많은 시간이 소모됩니다. 롤백 임계값을 낮추거나, K 값을 줄이세요.")
        
        if time_ratios.get('target_verification', 0) > 0.6:
            suggestions.append("타겟 모델 검증에 너무 많은 시간이 소모됩니다. 드래프트 모델의 품질을 개선하거나, 폴백 임계값을 조정하세요.")
        
        if time_ratios.get('overhead', 0) > 0.15:
            suggestions.append("오버헤드가 너무 큽니다. SSP 알고리즘 구현을 최적화하세요.")
        
        if not suggestions:
            suggestions.append("시간 분포가 적절하며, 현재 임계값 설정이 효율적입니다.")
        
        return {
            'efficiency': efficiency,
            'bottleneck': bottleneck,
            'ratios': time_ratios,
            'suggestions': suggestions
        }

    def _get_policy_parameters(self, states):
        """정책 네트워크에서 액션 분포 파라미터 계산"""
        action_params = self.policy_network(states)
        
        # 안정적인 액션 분포를 위한 파라미터 변환
        action_means = torch.zeros((states.shape[0], 2)).to(device)
        action_stds = torch.zeros((states.shape[0], 2)).to(device)
        
        # Fallback 파라미터 (0~1 범위)
        action_means[:, 0] = torch.tanh(action_params[:, 0])
        action_stds[:, 0] = torch.sigmoid(action_params[:, 1]) * 0.2 + 0.01
        
        # Rollback 파라미터 (0~k_value 범위)
        action_means[:, 1] = torch.tanh(action_params[:, 2])
        action_stds[:, 1] = torch.sigmoid(action_params[:, 3]) * 0.2 + 0.01
        
        return action_means, action_stds

def process_text_sample(text, target_model, draft_model, fallback_threshold, rollback_threshold, K, max_new_tokens):
    """텍스트 샘플 처리 - SSP 시간 통계 수집 및 프로세스 간 메모리 공유 이슈 해결"""
    try:
        # 워커 프로세스에서 CUDA 메모리 설정 재설정
        if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, '_set_allocator_settings'):
            torch.cuda.memory._set_allocator_settings('expandable_segments:False')
        
        # 시작 시간 기록
        start_time = time.time()
        
        # 인풋 인코딩
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]
        
        # SSP 실행 (시간 통계 수집)
        if HAS_MODIFIED_SSP:
            # SSP 함수 시그니처에 맞게 호출 - 순서 중요
            # ssp(target_model, draft_model, max_new_tokens, input_ids, K, ...)
            res = ssp(
                target_model=target_model, 
                draft_model=draft_model, 
                max_new_tokens=max_new_tokens,
                input_ids=inputs.input_ids,
                K=K, 
                fallback_threshold=fallback_threshold, 
                rollback_threshold=rollback_threshold,
                verbose=False
            )
            outputs, accepted_tokens, draft_tokens, time_stats = res
            
            # 수락률 계산 문제 수정
            # accepted_tokens는 수락된 토큰 수, draft_tokens는 생성된 총 토큰 수
            # 실제 수락률은 두 값의 비율이어야 함
            if draft_tokens > 0:
                # 수락률이 1.0(100%)보다 클 수 없도록 제한
                acceptance_rate = min(1.0, accepted_tokens / draft_tokens)
                logger.info(f"수락률 계산: {accepted_tokens}/{draft_tokens} = {acceptance_rate:.4f}")
            else:
                acceptance_rate = 0.0
                logger.warning("생성된 토큰이 없어 수락률을 0으로 설정")
        else:
            # 표준 SSP 실행 (시간 통계 없음)
            time_stats = {
                'draft_generation': 0.0,
                'target_verification': 0.0,
                'fallback_handling': 0.0,
                'rollback_handling': 0.0,
                'token_acceptance': 0.0,
                'overhead': 0.0
            }
            # 표준 SSP 함수 호출 - 명시적 인자 이름과 순서 사용
            # ssp(target_model, draft_model, max_new_tokens, input_ids, K, ...)
            outputs, accepted_tokens, draft_tokens, kl_divs = ssp(
                target_model=target_model, 
                draft_model=draft_model, 
                max_new_tokens=max_new_tokens,
                input_ids=inputs.input_ids,
                K=K, 
                fallback_threshold=fallback_threshold, 
                rollback_threshold=rollback_threshold,
                verbose=False
            )
            
            # 수락률 계산 문제 수정
            if draft_tokens > 0:
                # 수락률이 1.0(100%)보다 클 수 없도록 제한
                acceptance_rate = min(1.0, accepted_tokens / draft_tokens)
                logger.info(f"수락률 계산: {accepted_tokens}/{draft_tokens} = {acceptance_rate:.4f}")
            else:
                acceptance_rate = 0.0
                logger.warning("생성된 토큰이 없어 수락률을 0으로 설정")
        
        # 종료 시간 기록
        end_time = time.time()
        total_time = end_time - start_time
        
        # 결과 분석
        new_tokens = outputs.shape[1] - input_length
        if new_tokens <= 0:
            return {
                'success': False,
                'ms_per_token': None,
                'acceptance_rate': 0.0,
                'time_stats': time_stats
            }
        
        # 실행 시간 측정
        ms_per_token = (total_time * 1000) / new_tokens
        
        # 최종 결과 반환 - 수락률은 위에서 계산한 값 사용
        return {
            'success': True,
            'ms_per_token': ms_per_token,
            'acceptance_rate': acceptance_rate,
            'time_stats': time_stats
        }
    except Exception as e:
        # 에러는 상위 레벨로 전파하여 처리
        raise Exception(f"텍스트 처리 중 오류: {str(e)}")

def process_batch(texts, target_model, draft_model, state, K, max_new_tokens):
    """배치 단위로 텍스트 처리 - 상세 시간 통계 수집"""
    results = []
    batch_size = 1  # 배치 크기를 최소화하여 메모리 사용량 감소
    
    # 디버깅을 위한 정보 로깅
    logger.info(f"===== 배치 처리 시작 =====")
    logger.info(f"텍스트 수: {len(texts)}, K={K}, max_new_tokens={max_new_tokens}")
    logger.info(f"임계값 설정: fallback={state.fallback:.4f}, rollback={state.rollback:.4f}")
    
    # 모델 관련 디바이스 정보 확인
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
        if torch.cuda.is_available():
            target_embed_device = torch.device('cuda:0')
        else:
            target_embed_device = torch.device('cpu')
        logger.warning(f"타겟 모델 임베딩 장치를 찾을 수 없어 기본값 사용: {target_embed_device}")
    
    if draft_embed_device is None:
        # 타겟 모델과 같은 장치 사용
        draft_embed_device = target_embed_device
        logger.warning(f"드래프트 모델 임베딩 장치를 찾을 수 없어 타겟 장치 사용: {draft_embed_device}")
    
    # 장치 정보 로깅
    logger.info(f"모델 장치 - Target: {target_embed_device}, Draft: {draft_embed_device}")
    
    # 모델 정보 출력
    logger.info(f"타겟 모델 타입: {type(target_model).__name__}")
    logger.info(f"드래프트 모델 타입: {type(draft_model).__name__}")
    
    # 메모리 상태 확인
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            logger.info(f"CUDA 장치 {i} 메모리: {free_gb:.2f}GB 사용 가능 / {total_gb:.2f}GB 전체")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_results = []
        
        for text_idx, text in enumerate(batch_texts):
            try:
                logger.info(f"텍스트 {i + text_idx + 1}/{len(texts)} 처리 중...")
                
                # 텍스트 정보 로깅 (첫 100자만)
                text_preview = text[:100] + "..." if len(text) > 100 else text
                logger.info(f"텍스트 미리보기: {text_preview}")
                
                # 토크나이저 결과를 타겟 모델의 임베딩 레이어와 동일한 장치로 이동
                logger.info(f"토큰화 시작...")
                try:
                    input_tensor = tokenizer(
                        text, 
                        return_tensors="pt", 
                        max_length=512,  
                        truncation=True  
                    )
                    
                    # 장치 이동
                    input_ids = input_tensor.input_ids.to(target_embed_device)
                    logger.info(f"토큰화 완료: {input_ids.shape} (장치: {input_ids.device})")
                except Exception as e:
                    logger.error(f"토큰화 오류: {str(e)}")
                    raise
                
                start_time = time.time()
                
                # SSP 함수 호출 전 정보 로깅
                logger.info(f"SSP 시작: fallback={state.fallback:.4f}, rollback={state.rollback:.4f}, K={K}")
                
                # ssp_modified 함수 호출 (시간 통계 포함)
                try:
                    # verbose 모드 활성화
                    # 시간 통계를 반환하는 ssp 함수 호출
                    from lssp.ssp_modified import ssp as ssp_modified
                    logger.info("ssp_modified 함수 호출 중...")
                    
                    # 실제 사용 시에는 더 많은 토큰 생성 시도 (16 -> 32)
                    actual_max_tokens = min(max_new_tokens, 32)  # 메모리 부족 방지를 위해 최대값 제한
                    
                    # 빠른 테스트를 위해 작은 값으로 설정
                    if len(texts) <= 2: # 테스트 모드
                        actual_max_tokens = min(max_new_tokens, 16)  # 테스트 모드에서는 적은 토큰 생성
                    
                    generated_ids, accept_tokens, generated_tokens, time_stats = ssp_modified(
                        target_model=target_model,
                        draft_model=draft_model,
                        max_new_tokens=actual_max_tokens,
                        input_ids=input_ids,
                        K=K,
                        fallback_threshold=state.fallback,
                        rollback_threshold=state.rollback,
                        verbose=True  # 디버깅을 위해 verbose 모드 활성화
                    )
                    logger.info(f"ssp_modified 함수 호출 완료")
                except (TypeError, ValueError, ImportError) as e:
                    # 기존 ssp 함수는 시간 통계를 반환하지 않을 수 있음
                    logger.warning(f"ssp_modified 함수 호출 실패, 기본 ssp 함수 사용: {str(e)}")
                    try:
                        # 기본 ssp 함수 가져오기
                        from lssp.ssp import ssp as ssp_original
                        logger.info("기본 ssp 함수 호출 중...")
                        # 기본 ssp 함수는 3개의 값만 반환
                        generated_ids, accept_tokens, generated_tokens = ssp_original(
                            target_model,
                            draft_model,
                            max_new_tokens,
                            input_ids,
                            K=K,
                            fallback_threshold=state.fallback,
                            rollback_threshold=state.rollback
                        )
                        logger.info(f"기본 ssp 함수 호출 완료")
                    except Exception as inner_e:
                        # 다른 오류 발생 시 로깅
                        logger.error(f"ssp 함수 호출 중 오류 발생: {str(inner_e)}")
                        import traceback
                        logger.error(f"오류 상세: {traceback.format_exc()}")
                        raise
                        
                    # 기본 시간 통계 생성
                    time_stats = {
                        'draft_generation': 0.0,
                        'target_verification': 0.0,
                        'fallback_handling': 0.0,
                        'rollback_handling': 0.0,
                        'token_acceptance': 0.0,
                        'overhead': 0.0
                    }
                
                elapsed = time.time() - start_time
                
                # 결과 확인 및 로깅
                if generated_ids is None:
                    logger.error("SSP 결과가 None입니다!")
                    raise ValueError("SSP 결과가 None입니다")
                    
                logger.info(f"SSP 완료: 입력 크기={input_ids.shape}, 출력 크기={generated_ids.shape}")
                
                num_new_tokens = generated_ids.shape[1] - input_ids.shape[1]
                logger.info(f"생성된 새 토큰 수: {num_new_tokens}")
                
                if num_new_tokens > 0:
                    ms_per_token = (elapsed * 1000) / num_new_tokens
                else:
                    ms_per_token = 1000.0
                    logger.warning("새 토큰이 생성되지 않았습니다")
                    
                acceptance_rate = accept_tokens / max(generated_tokens, 1)
                
                logger.info(f"생성 통계: {accept_tokens}/{generated_tokens} 토큰 수락 ({acceptance_rate*100:.2f}%), {ms_per_token:.2f}ms/토큰")
                logger.info(f"시간 통계: {time_stats}")
                
                # 결과에 시간 통계 포함
                batch_results.append({
                    'ms_per_token': ms_per_token,
                    'acceptance_rate': acceptance_rate,
                    'time_stats': time_stats,
                    'success': True
                })
                
                # 메모리 해제
                del input_ids, generated_ids
                # 명시적 가비지 컬렉션
                if i % 5 == 0:  # 5개 배치마다 가비지 컬렉션 실행
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"텍스트 처리 중 오류 발생: {str(e)}")
                # 스택 트레이스 출력
                import traceback
                logger.error(f"오류 상세: {traceback.format_exc()}")
                
                # 기본 결과 생성
                batch_results.append({
                    'ms_per_token': 1000.0,
                    'acceptance_rate': 0.0,
                    'time_stats': {
                        'draft_generation': 0.0,
                        'target_verification': 0.0,
                        'fallback_handling': 0.0,
                        'rollback_handling': 0.0,
                        'token_acceptance': 0.0,
                        'overhead': 0.0
                    },
                    'success': False
                })
        
        results.extend(batch_results)
    
    # 처리 완료 후 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 성공/실패 분석
    success_count = sum(1 for r in results if r.get('success', False))
    logger.info(f"배치 처리 완료: {success_count}/{len(results)} 성공")
    
    # 평균 계산 (성공한 결과만 사용)
    successful_results = [r for r in results if r.get('success', False)]
    if successful_results:
        avg_ms = sum(r['ms_per_token'] for r in successful_results) / len(successful_results)
        avg_acc = sum(r['acceptance_rate'] for r in successful_results) / len(successful_results)
        logger.info(f"평균 성능 (성공만): {avg_ms:.2f}ms/토큰, 수락률={avg_acc*100:.2f}%")
    
    return results

def evaluate_thresholds_parallel(target_model, draft_model, state, texts, K=16, max_new_tokens=64, episode=0):
    """임계값 설정에 대한 병렬 평가 - 레이턴시 중심 최적화 및 안정성 개선"""
    eval_start_time = time.time()
    
    # 텍스트 샘플링 (최대 10개만 사용 - 평가 속도 향상)
    eval_texts = random.sample(texts, min(10, len(texts)))
    
    # 병렬 처리 수행
    logger.info(f"{'='*10} 임계값 평가 시작: FB={state.fallback:.3f}, RB={state.rollback:.2f} {'='*10}")
    
    # 병렬 처리 대신 직렬 처리로 변경 (CUDA 메모리 공유 문제 해결)
    try:
        all_results = []
        for i, text in enumerate(eval_texts):
            try:
                result = process_text_sample(
                    text, target_model, draft_model, 
                    state.fallback, state.rollback, K, max_new_tokens
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"텍스트 {i} 처리 실패: {str(e)}")
    except Exception as e:
        logger.error(f"처리 실패: {str(e)}")
        # 기본 결과 반환 (평가 실패)
        return -10.0, {'ms_per_token': 1000.0, 'acceptance_rate': 0.0, 'reward': -10.0}
    
    # 유효한 결과만 필터링
    valid_results = [r for r in all_results if r and r.get('ms_per_token') is not None]
    
    if not valid_results:
        logger.warning("유효한 평가 결과 없음")
        return -10.0, {'ms_per_token': 1000.0, 'acceptance_rate': 0.0, 'reward': -10.0}
    
    # 평균 통계 계산
    avg_ms_per_token = sum(r.get('ms_per_token', 0) for r in valid_results) / len(valid_results)
    avg_acceptance_rate = sum(r.get('acceptance_rate', 0) for r in valid_results) / len(valid_results)
    
    # 시간 통계 평균 계산
    time_keys = ['draft_generation', 'target_verification', 'fallback_handling', 
                'rollback_handling', 'token_acceptance', 'overhead']
    avg_time_stats = {key: 0.0 for key in time_keys}
    
    valid_time_stats = []
    for r in valid_results:
        if 'time_stats' in r and r['time_stats']:
            valid_time_stats.append(r['time_stats'])
    
    if valid_time_stats:
        for key in time_keys:
            avg_time_stats[key] = sum(stats.get(key, 0.0) for stats in valid_time_stats) / len(valid_time_stats)
    
    # 레이턴시 중심 보상 계산
    reward = compute_reward(
        avg_acceptance_rate, 
        avg_ms_per_token,
        time_stats=avg_time_stats,
        min_acceptance=0.0,  # 수락률 임계값 제거
        episode=episode,
        fallback_value=state.fallback,
        rollback_value=state.rollback,
        k_value=K
    )
    
    # 결과 로깅
    logger.info(f"평가 결과: 레이턴시={avg_ms_per_token:.2f}ms/token, 수락률={avg_acceptance_rate*100:.1f}%, 보상={reward:.2f}")
    
    # 메트릭스 반환
    metrics = {
        'ms_per_token': avg_ms_per_token,
        'acceptance_rate': avg_acceptance_rate,
        'reward': reward,
        'time_stats': avg_time_stats
    }
    
    return reward, metrics

def train_model_pair(target_name, draft_name, agent, texts, episodes=20, K=16, max_new_tokens=64):
    """특정 모델 쌍에 대한 임계값 최적화 훈련 - 레이턴시 최적화 중심 및 안정성 개선"""
    max_retries = 3
    retry_delay = 5  # seconds
    
    try:
        # 모델 로드
        target_params = models_params[target_name].copy()
        draft_params = models_params[draft_name].copy()
        
        target_model = create_model_wrapper(**target_params)
        draft_model = create_model_wrapper(**draft_params)
    except Exception as e:
        logger.error(f"모델 로드 실패: {str(e)}")
        return []
    
    results = []
    
    # 임계값 범위 내에서 초기 상태 설정
    fb_init = 0.2  # 안정적인 시작값
    rb_init = 3.0  # 안정적인 시작값
    state = ThresholdState(fallback=fb_init, rollback=rb_init)
    
    # 탐색 전략
    exploration_rate = 0.8  # 초기 탐색률 (약간 감소)
    exploration_decay = 0.95  # 느린 감소율
    
    # 가장 좋은 결과 추적
    best_ms_per_token = float('inf')
    best_state = None
    best_reward = float('-inf')
    
    # 이전 에피소드들의 결과 저장 (moving average 계산용)
    latency_history = []
    
    for episode in range(episodes):
        logger.info(f"에피소드 {episode+1}/{episodes} 시작 - 모델: {target_name}/{draft_name}")
        
        # 에피소드별 탐색 전략
        # 초반에는 다양한 범위 탐색, 후반에는 개선된 영역 집중 탐색
        if episode < episodes // 3:
            # 초반: 넓은 영역 탐색
            explore_threshold = 0.9
            explore_range_fallback = (0.05, 0.4)
            explore_range_rollback = (1.0, min(10.0, K))
        elif episode < episodes * 2 // 3:
            # 중반: 적당한 영역 탐색
            explore_threshold = 0.7
            
            # 지금까지 좋은 결과 기반으로 탐색 범위 좁히기
            if best_state:
                fb_center = best_state.fallback
                rb_center = best_state.rollback
                explore_range_fallback = (max(0.05, fb_center - 0.15), min(0.5, fb_center + 0.15))
                explore_range_rollback = (max(1.0, rb_center - 2.0), min(K, rb_center + 2.0))
            else:
                explore_range_fallback = (0.1, 0.3)
                explore_range_rollback = (1.0, min(6.0, K))
        else:
            # 후반: 미세 조정
            explore_threshold = 0.5
            
            # 더 좁은 범위 탐색
            if best_state:
                fb_center = best_state.fallback
                rb_center = best_state.rollback
                explore_range_fallback = (max(0.05, fb_center - 0.1), min(0.5, fb_center + 0.1))
                explore_range_rollback = (max(1.0, rb_center - 1.0), min(K, rb_center + 1.0))
            else:
                explore_range_fallback = (0.1, 0.3)
                explore_range_rollback = (2.0, min(6.0, K))
        
        # 액션 결정
        if random.random() < exploration_rate and random.random() < explore_threshold:
            # 랜덤 탐색 (설정된 범위 내에서)
            next_fallback = random.uniform(explore_range_fallback[0], explore_range_fallback[1])
            next_rollback = random.uniform(explore_range_rollback[0], explore_range_rollback[1])
            next_state = ThresholdState(fallback=next_fallback, rollback=next_rollback)
            old_log_prob = 0.0
        else:
            # 정책 기반 액션
            action_values, old_log_prob = agent.get_action(state, explore=(random.random() < exploration_rate))
            next_state = ThresholdState(fallback=action_values[0], rollback=action_values[1])
        
        # 탐색률 감소
        exploration_rate *= exploration_decay
        
        # 평가 수행 (재시도 로직 포함)
        reward = None
        metrics = None
        
        for eval_attempt in range(max_retries):
            try:
                reward, metrics = evaluate_thresholds_parallel(
                    target_model, draft_model, next_state, texts, K, max_new_tokens, episode
                )
                break
            except Exception as e:
                if eval_attempt < max_retries - 1:
                    logger.warning(f"평가 실패: {str(e)}. {retry_delay}초 후 재시도...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"평가 최종 실패: {str(e)}")
                    # 실패 시 안전한 기본값 사용
                    metrics = {
                        'ms_per_token': 1000.0,
                        'acceptance_rate': 0.0,
                        'reward': -10.0
                    }
                    reward = -10.0
                    break
        
        # 결과 기록
        latency_history.append(metrics['ms_per_token'])
        if len(latency_history) > 5:  # 최근 5개 에피소드만 유지
            latency_history.pop(0)
        
        # 최고 성능 업데이트
        if metrics['ms_per_token'] < best_ms_per_token:
            best_ms_per_token = metrics['ms_per_token']
            best_state = next_state
            best_reward = reward
            logger.info(f"새로운 최고 레이턴시: {best_ms_per_token:.2f}ms/token, FB={next_state.fallback:.2f}, RB={next_state.rollback:.1f}")
        
        # 경험 버퍼에 추가
        agent.experience_buffer.add(
            Experience(
                state, 
                (next_state.fallback - state.fallback, next_state.rollback - state.rollback), 
                reward, 
                next_state, 
                metrics, 
                old_log_prob
            )
        )
        
        # 정책 업데이트
        if episode > 0 and episode % 2 == 0:  # 2 에피소드마다 업데이트
            agent.update()
        
        # 결과 기록
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
        
        # 시간 통계 추가
        if 'time_stats' in metrics:
            result['time_stats'] = metrics['time_stats']
        
        results.append(result)
        
        # 상태 업데이트
        state = next_state
        
        # 이동 평균 출력
        avg_latency = sum(latency_history) / len(latency_history)
        logger.info(f"에피소드 {episode+1} 완료: 레이턴시={metrics['ms_per_token']:.2f}ms/token, 평균={avg_latency:.2f}ms/token")
        
        # 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 최종 결과 요약
    if best_state:
        logger.info(f"최종 최적 임계값: FB={best_state.fallback:.2f}, RB={best_state.rollback:.1f}")
        logger.info(f"최고 레이턴시: {best_ms_per_token:.2f}ms/token")
    
    return results

def meta_learning(datasets, model_pairs, args):
    """다양한 모델/데이터셋에 대한 메타러닝 - 레이턴시 중심 최적화"""
    
    # K 값 전달하여 PPOAgent 초기화
    agent = PPOAgent(k_value=args.k)
    
    all_results = []
    best_thresholds = {}
    best_latencies = {}
    time_stats_by_model = {}
    
    logger.info(f"메타러닝 시작: {len(datasets)} 데이터셋, {len(model_pairs)} 모델 쌍")
    logger.info(f"학습 설정: K={args.k}, 에피소드={args.episodes_per_pair}, 토큰={args.max_tokens}")
    
    for dataset_name, texts in datasets.items():
        logger.info(f"\n{'='*30}\n데이터셋 {dataset_name}에 대한 훈련 시작\n{'='*30}")
        
        for target_name, draft_name in model_pairs:
            model_key = f"{target_name}_{draft_name}_{dataset_name}"
            logger.info(f"\n{'='*20} 모델 쌍 {target_name}/{draft_name} 훈련 {'='*20}")
            
            # 모델 쌍 훈련
            pair_results = train_model_pair(
                target_name, draft_name, agent, texts, 
                episodes=args.episodes_per_pair,
                K=args.k,
                max_new_tokens=args.max_tokens
            )
            
            if not pair_results:
                logger.warning(f"모델 쌍 {target_name}/{draft_name}에 대한 결과가 없습니다.")
                continue
                
            # 레이턴시 기준으로 최적 결과 선택 (ms_per_token이 가장 낮은 것)
            best_result = min(pair_results, key=lambda x: x['ms_per_token'])
            
            # 이 모델 쌍에 대한 최고 레이턴시 저장
            best_latencies[model_key] = best_result['ms_per_token']
            
            # 최적 임계값 저장
            best_thresholds[model_key] = {
                'fallback': best_result['fallback_threshold'],
                'rollback': best_result['rollback_threshold'],
                'acceptance_rate': best_result['acceptance_rate'],
                'ms_per_token': best_result['ms_per_token'],
                'reward': best_result['reward']
            }
            
            # 시간 통계가 있는 경우 추가
            if 'time_stats' in best_result:
                best_thresholds[model_key]['time_stats'] = best_result['time_stats']
                time_stats_by_model[model_key] = best_result['time_stats']
            
            # 훈련 완료 후 최적 결과 출력
            logger.info(f"\n최적 임계값 (레이턴시 기준):")
            logger.info(f"FB={best_result['fallback_threshold']:.3f}, RB={best_result['rollback_threshold']:.2f}")
            logger.info(f"레이턴시={best_result['ms_per_token']:.2f}ms/token, 수락률={best_result['acceptance_rate']*100:.1f}%")
            
            # 모든 결과 기록
            for result in pair_results:
                result['dataset'] = dataset_name
                all_results.append(result)
    
    # 모델 저장
    agent.save_model(args.model_output)
    logger.info(f"모델 저장 완료: {args.model_output}")
    
    # 학습 이력 그래프
    try:
        agent.plot_training_history(args.model_output.replace('.pt', '_history.png'))
        logger.info(f"학습 이력 그래프 저장 완료")
    except Exception as e:
        logger.error(f"학습 이력 그래프 생성 실패: {str(e)}")
    
    # 모든 최적 임계값 요약 출력
    logger.info("\n모든 모델 쌍에 대한 최적 임계값 요약:")
    for model_key, thresholds in best_thresholds.items():
        logger.info(f"{model_key}: FB={thresholds['fallback']:.3f}, RB={thresholds['rollback']:.2f}, " + 
                   f"레이턴시={thresholds['ms_per_token']:.2f}ms")
    
    # 결과 저장
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'best_thresholds': best_thresholds,
                'best_latencies': best_latencies
            }, f, indent=2)
        logger.info(f"결과 저장 완료: {args.output}")
        
        # 시간 통계 저장
        time_stats_path = args.output.replace('.json', '_time_stats.json')
        with open(time_stats_path, 'w') as f:
            json.dump(time_stats_by_model, f, indent=2)
        logger.info(f"시간 통계 저장 완료: {time_stats_path}")
    
    return all_results, best_thresholds, agent, time_stats_by_model

def cross_validation(agent, datasets, model_pairs, args):
    """학습된 에이전트 크로스 검증 - 시간 통계 활용"""
    
    validation_results = []
    time_stats_by_model = {}
    
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
            init_state = ThresholdState(fallback=0.1, rollback=2.0)
            
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
            
            # 시간 통계 추가
            if 'time_stats' in metrics:
                result['time_stats'] = metrics['time_stats']
                
                # 모델별 시간 통계 수집
                model_key = f"{target_name}_{draft_name}_{dataset_name}"
                if model_key not in time_stats_by_model:
                    time_stats_by_model[model_key] = []
                time_stats_by_model[model_key].append(metrics['time_stats'])
                
                # 시간 통계 출력
                time_stats = metrics['time_stats']
                time_stats_str = ", ".join([f"{k[:5]}:{v*1000:.1f}ms" for k, v in time_stats.items()])
                logger.info(f"시간 통계: {time_stats_str}")
            
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
    
    # 시간 통계 보고서 작성
    time_stats_report = {}
    for model_key, stats_list in time_stats_by_model.items():
        if not stats_list:
            continue
            
        avg_stats = {}
        for stat_key in stats_list[0].keys():
            avg_stats[stat_key] = sum(stats[stat_key] for stats in stats_list) / len(stats_list)
        
        time_stats_report[model_key] = avg_stats
    
    return validation_results, time_stats_report

def main():
    parser = argparse.ArgumentParser(description="Advanced RL for threshold optimization with time statistics")
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
    parser.add_argument('--analyze-time-stats', action='store_true',
                        help='Perform detailed time statistics analysis')
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
    all_results, best_thresholds, agent, time_stats_training = meta_learning(datasets, model_pairs, args)
    
    # 크로스 검증
    print(f"\n{'='*30} 크로스 검증 시작 {'='*30}")
    validation_results, time_stats_validation = cross_validation(agent, datasets, model_pairs, args)
    
    # 결과 저장
    final_results = {
        'training': all_results,
        'best_thresholds': best_thresholds,
        'validation': validation_results,
        'time_stats_training': time_stats_training,
        'time_stats_validation': time_stats_validation
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
        print(f"  - 보상: {thresholds['reward']:.2f}")
        
        # 시간 통계 요약 (있는 경우)
        if 'time_stats' in thresholds:
            time_stats = thresholds['time_stats']
            total_time = sum(time_stats.values())
            if total_time > 0:
                proportions = {k: (v/total_time) * 100 for k, v in time_stats.items()}
                print(f"  - 시간 분석: draft:{proportions.get('draft_generation', 0):.1f}%, target:{proportions.get('target_verification', 0):.1f}%, "
                      f"FB:{proportions.get('fallback_handling', 0):.1f}%, RB:{proportions.get('rollback_handling', 0):.1f}%")
            else:
                print("  - 시간 분석: 유효한 시간 통계 없음")
    
    # 검증 결과 요약
    print("\n크로스 검증 결과:")
    for result in validation_results:
        print(f"* {result['target']}/{result['draft']} on {result['dataset']}:")
        print(f"  추천: FB={result['fallback_threshold']:.3f}, RB={result['rollback_threshold']:.3f}, "
              f"수락률={result['acceptance_rate']:.3f}, 속도={result['ms_per_token']:.2f}ms/token")
        
        # 시간 통계 요약 (있는 경우)
        if 'time_stats' in result:
            time_stats = result['time_stats']
            total_time = sum(time_stats.values())
            if total_time > 0:
                proportions = {k: (v/total_time) * 100 for k, v in time_stats.items()}
                print(f"  시간 분석: draft:{proportions.get('draft_generation', 0):.1f}%, target:{proportions.get('target_verification', 0):.1f}%, "
                      f"FB:{proportions.get('fallback_handling', 0):.1f}%, RB:{proportions.get('rollback_handling', 0):.1f}%")
            else:
                print("  시간 분석: 유효한 시간 통계 없음")
    
    # 일반화된 추천값
    print("\n일반화된 추천 임계값 (모든 모델에 적용 가능):")
    avg_fallback = sum(r['fallback_threshold'] for r in validation_results) / len(validation_results)
    avg_rollback = sum(r['rollback_threshold'] for r in validation_results) / len(validation_results)
    print(f"Fallback: {avg_fallback:.3f}, Rollback: {avg_rollback:.3f}")
    
    # 시간 통계 분석 (옵션)
    if args.analyze_time_stats:
        print("\n" + "=" * 50)
        print("시간 통계 상세 분석:")
        print("-" * 50)
        
        # 각 모델 쌍별 시간 통계 분석
        all_time_stats = {**time_stats_training, **time_stats_validation}
        
        for model_key, stats in all_time_stats.items():
            print(f"\n* {model_key} 시간 분석:")
            total_time = sum(stats.values())
            
            if total_time <= 0:
                print("  - 유효한 시간 통계 없음")
                continue
                
            # 각 단계별 시간 비율
            for step, time_value in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (time_value / total_time) * 100
                ms_per_step = time_value * 1000
                print(f"  - {step}: {ms_per_step:.2f}ms ({percentage:.1f}%)")
            
            # 시간 효율성 지표 계산
            if 'draft_generation' in stats and 'target_verification' in stats:
                draft_time = stats['draft_generation']
                target_time = stats['target_verification']
                if target_time > 0:
                    draft_target_ratio = draft_time / target_time
                    print(f"  - Draft/Target 비율: {draft_target_ratio:.2f}")
                    if draft_target_ratio < 0.5:
                        print("    => 드래프트 모델이 효율적으로 동작하고 있습니다.")
                    elif draft_target_ratio > 1.0:
                        print("    => 드래프트 모델이 너무 느립니다. 더 작은 모델이나 최적화된 모델을 고려하세요.")
                else:
                    print("  - Draft/Target 비율: 계산할 수 없음 (Target 시간이 0)")
            
            # 폴백/롤백 오버헤드 분석
            if 'fallback_handling' in stats and 'rollback_handling' in stats:
                fb_time = stats['fallback_handling']
                rb_time = stats['rollback_handling']
                fb_rb_overhead = (fb_time + rb_time) / total_time * 100
                
                print(f"  - 폴백/롤백 오버헤드: {fb_rb_overhead:.1f}%")
                if fb_rb_overhead > 25:
                    print("    => 폴백/롤백 처리가 너무 많은 시간을 소모합니다. 임계값 조정이 필요합니다.")
                elif fb_rb_overhead < 10:
                    print("    => 폴백/롤백 처리가 효율적입니다.")
    
    print("=" * 50)
    
    # 최적의 임계값 추천
    if args.optimize:
        print("\n최적 임계값 계산 중...")
        for k in [8, 12, 16, 24, 32]:
            try:
                k_agent = PPOAgent(k_value=k)
                k_agent.load_model(f"models/ppo_agent_k{k}.pt")
                
                # 초기 상태를 K 값에 맞게 조정
                init_state = ThresholdState(fallback=0.1, rollback=2.0)
                action_values, _ = k_agent.get_action(init_state)
                
                # 새로운 상태 생성
                new_fallback = max(0.0, min(1.0, action_values[0]))
                new_rollback = max(0.0, min(min(k, 16.0), action_values[1]))
                recommended_state = ThresholdState(new_fallback, new_rollback)
                
                # K를 초과하지 않도록 제한
                recommended_state.rollback = min(recommended_state.rollback, k)
                
                print(f"K={k}: 폴백 임계값={recommended_state.fallback:.4f}, 롤백 임계값={recommended_state.rollback:.4f}")
            except Exception as e:
                print(f"K={k}에 대한 모델을 로드할 수 없습니다: {str(e)}")

if __name__ == "__main__":
    main() 
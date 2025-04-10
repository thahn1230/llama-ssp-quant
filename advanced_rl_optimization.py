import os
import argparse
import torch
import time
import json
import logging
import numpy as np
import random
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.multiprocessing as mp

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# llamassp 모듈에서 필요한 함수들 임포트
from llamassp import create_model, tokenizer, models_params, MAX_NEW_TOKENS
from lssp.ssp import ssp
from lssp.base import sample_model

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 파라미터 정의
models_params = {
    "13B": {
        "model_name": "facebook/opt-6.7b",
        "max_memory": {0: "22GB", 1: "22GB", 2: "22GB", 3: "22GB", 4: "22GB", 5: "22GB", 6: "22GB", 7: "22GB"},
        "device_map": "balanced",
        "offload_folder": "offload_folder"
    },
    "7B": {
        "model_name": "facebook/opt-1.3b",
        "max_memory": {0: "22GB", 1: "22GB", 2: "22GB", 3: "22GB", 4: "22GB", 5: "22GB", 6: "22GB", 7: "22GB"},
        "device_map": "balanced"
    },
    "3B": {
        "model_name": "facebook/opt-125m",
        "max_memory": {0: "22GB", 1: "22GB", 2: "22GB", 3: "22GB", 4: "22GB", 5: "22GB", 6: "22GB", 7: "22GB"},
        "device_map": "balanced"
    }
}

def create_model_wrapper(**kwargs):
    """
    create_model을 래핑하는 함수
    """
    model_kwargs = kwargs.copy()
    logger.info(f"모델 로드 중: {model_kwargs.get('model_name', '알 수 없음')}")
    
    # create_model이 지원하지 않는 매개변수 제거
    for key in ['torch_dtype', 'low_cpu_mem_usage', 'load_in_4bit', 'load_in_8bit', 'quantization_config']:
        model_kwargs.pop(key, None)
    
    # create_model 함수 호출
    return create_model(**model_kwargs)

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

def compute_reward(acceptance_rate, ms_per_token, speed_weight=0.4, acceptance_weight=0.6, 
                  baseline_ms=100.0, min_acceptance=0.3, episode=0):
    """개선된 보상 함수"""
    if np.isnan(acceptance_rate) or np.isnan(ms_per_token):
        return 0.0  # 기본값 반환
    
    if acceptance_rate < min_acceptance:
        penalty = -((min_acceptance - acceptance_rate) * 3.0) ** 2
        return penalty
    
    speed_score = baseline_ms / max(ms_per_token, 1.0)
    speed_score = np.tanh(speed_score - 1.0) + 1.0
    acceptance_score = acceptance_rate ** 0.4
    
    episode_progress = min(1.0, episode / 100)
    acceptance_weight = 0.8 - (0.2 * episode_progress)
    speed_weight = 0.2 + (0.2 * episode_progress)
    
    reward = (speed_weight * speed_score) + (acceptance_weight * acceptance_score)
    return reward

class PPOAgent:
    """Proximal Policy Optimization 알고리즘 구현"""
    
    def __init__(self, fallback_range=(0.05, 0.95), rollback_range=(1.0, 50.0),
                 lr=0.001, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, 
                 train_iterations=10, batch_size=64):
        self.fallback_range = fallback_range
        self.rollback_range = rollback_range
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.train_iterations = train_iterations
        self.batch_size = batch_size
        
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
        """현재 상태에서 액션 선택"""
        state_tensor = torch.FloatTensor([state.fallback, state.rollback]).to(device)  # 장치로 이동
        
        with torch.no_grad():
            action_params = self.policy_network(state_tensor)
            
            fallback_mean = torch.tanh(action_params[0])  # -1 ~ 1
            fallback_std = torch.sigmoid(action_params[1]) * 0.5  # 0 ~ 0.5
            rollback_mean = torch.tanh(action_params[2])
            rollback_std = torch.sigmoid(action_params[3]) * 0.5
        
        try:
            if explore:
                delta_fallback = torch.normal(mean=fallback_mean, std=fallback_std).item() * 0.1
                delta_rollback = torch.normal(mean=rollback_mean, std=rollback_std).item() * 2.0
            else:
                delta_fallback = fallback_mean.item() * 0.1
                delta_rollback = rollback_mean.item() * 2.0
        except Exception as e:
            print(f"액션 샘플링 오류: {e}, 기본값 사용")
            delta_fallback = 0.05 if explore else 0.0
            delta_rollback = 1.0 if explore else 0.0
        
        new_fallback = np.clip(state.fallback + delta_fallback, 
                              self.fallback_range[0], self.fallback_range[1])
        new_rollback = np.clip(state.rollback + delta_rollback, 
                              self.rollback_range[0], self.rollback_range[1])
        
        action_params = {
            'fallback': (fallback_mean, fallback_std),
            'rollback': (rollback_mean, rollback_std)
        }
        
        return ThresholdState(new_fallback, new_rollback), action_params
    
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
        
        advantages, returns = self.compute_gae(rewards, values, next_value, dones)
        
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
            
            fallback_means = torch.tanh(action_params[:, 0])
            fallback_stds = torch.sigmoid(action_params[:, 1]) * 0.5
            rollback_means = torch.tanh(action_params[:, 2])
            rollback_stds = torch.sigmoid(action_params[:, 3]) * 0.5
            
            if torch.isnan(fallback_means).any() or torch.isnan(fallback_stds).any() or \
               torch.isnan(rollback_means).any() or torch.isnan(rollback_stds).any():
                logger.warning("NaN detected in action parameters. Skipping update.")
                return
            
            fallback_dist = torch.distributions.Normal(fallback_means, fallback_stds)
            rollback_dist = torch.distributions.Normal(rollback_means, rollback_stds)
            
            fallback_log_probs = fallback_dist.log_prob(actions[:, 0])
            rollback_log_probs = rollback_dist.log_prob(actions[:, 1])
            
            new_log_probs = fallback_log_probs + rollback_log_probs
            
            old_log_probs = torch.FloatTensor([exp.old_log_prob for exp in batch]).to(device)  # 장치로 이동
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            clip_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            
            policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
            
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            values = self.value_network(states).squeeze()
            value_loss = torch.nn.MSELoss()(values, returns)
            
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
            self.value_optimizer.step()
            
            self.training_history['policy_loss'].append(policy_loss.item())
            self.training_history['value_loss'].append(value_loss.item())
            self.training_history['advantages'].append(advantages.mean().item())
        
        self.training_history['rewards'].extend(rewards.tolist())
        
        if len(self.experience_buffer) > self.batch_size * 2:
            self.experience_buffer.buffer = deque(list(self.experience_buffer.buffer)[-self.batch_size:])
    
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

def process_batch(texts, target_model, draft_model, state, K, max_new_tokens):
    """배치 단위로 텍스트 처리"""
    results = []
    batch_size = 4  # 배치 크기
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_results = []
        
        for text in batch_texts:
            try:
                input_ids = tokenizer(
                    text, 
                    return_tensors="pt", 
                    max_length=512,  
                    truncation=True  
                ).input_ids.to(target_model.device)  # 모델의 장치로 이동
                
                start_time = time.time()
                
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
            except Exception as e:
                logger.warning(f"텍스트 처리 중 오류 발생: {str(e)}")
                batch_results.append({
                    'ms_per_token': 1000.0,
                    'acceptance_rate': 0.0
                })
        
        results.extend(batch_results)
    
    return results

def evaluate_thresholds_parallel(target_model, draft_model, state, texts, K=16, max_new_tokens=128, episode=0):
    """병렬 처리로 임계값 평가"""
    num_processes = min(4, len(texts))  # 최대 4개 프로세스 사용
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
    
    reward = compute_reward(avg_acceptance_rate, avg_ms_per_token, episode=episode)
    
    metrics = {
        'ms_per_token': avg_ms_per_token,
        'acceptance_rate': avg_acceptance_rate,
        'reward': reward
    }
    
    logger.info(f"평가 결과 ({state}): 수락률={avg_acceptance_rate:.3f}, 속도={avg_ms_per_token:.2f}ms/token, 보상={reward:.3f}")
    
    return reward, metrics

def train_model_pair(target_name, draft_name, agent, texts, episodes=20, K=16, max_new_tokens=128):
    """특정 모델 쌍에 대한 에이전트 훈련"""
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"모델 로드 중: {target_name} (타겟), {draft_name} (드래프트) - 시도 {attempt + 1}/{max_retries}")
            target_model = create_model_wrapper(**models_params[target_name]).to(device)  # 장치로 이동
            draft_model = create_model_wrapper(**models_params[draft_name]).to(device)  # 장치로 이동
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"모델 로드 실패: {str(e)}. {retry_delay}초 후 재시도...")
                time.sleep(retry_delay)
            else:
                logger.error(f"모델 로드 최종 실패: {str(e)}")
                raise
    
    results = []
    
    state = ThresholdState(fallback=0.4, rollback=15.0)
    
    exploration_rate = 1.0
    exploration_decay = 0.95
    
    for episode in range(episodes):
        logger.info(f"에피소드 {episode+1}/{episodes} 시작 - 모델: {target_name}/{draft_name}")
        
        if random.random() < exploration_rate:
            next_state = ThresholdState(
                fallback=random.uniform(0.3, 0.5),
                rollback=random.uniform(10.0, 20.0)
            )
            action_params = None
        else:
            next_state, action_params = agent.get_action(state)
        
        exploration_rate *= exploration_decay
        
        if action_params is not None:
            with torch.no_grad():
                fallback_dist = torch.distributions.Normal(
                    action_params['fallback'][0],
                    action_params['fallback'][1]
                )
                rollback_dist = torch.distributions.Normal(
                    action_params['rollback'][0],
                    action_params['rollback'][1]
                )
                
                delta_fallback = (next_state.fallback - state.fallback) / 0.1
                delta_rollback = (next_state.rollback - state.rollback) / 2.0
                
                old_log_prob = (
                    fallback_dist.log_prob(torch.tensor(delta_fallback).to(device)) +
                    rollback_dist.log_prob(torch.tensor(delta_rollback).to(device))
                ).item()
        else:
            old_log_prob = 0.0
        
        for eval_attempt in range(max_retries):
            try:
                reward, metrics = evaluate_thresholds_parallel(
                    target_model, draft_model, next_state, texts, K, max_new_tokens, episode
                )
                reward = compute_reward(
                    metrics['acceptance_rate'], 
                    metrics['ms_per_token'],
                    episode=episode
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
    
    return results

def meta_learning(datasets, model_pairs, args):
    """다양한 모델/데이터셋에 대한 메타러닝 수행"""
    
    agent = PPOAgent()
    
    all_results = []
    best_thresholds = {}
    
    for dataset_name, texts in datasets.items():
        logger.info(f"데이터셋 {dataset_name}에 대한 훈련 시작")
        
        for target_name, draft_name in model_pairs:
            logger.info(f"모델 쌍 {target_name}/{draft_name} 훈련 시작")
            
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
            
            for result in pair_results:
                result['dataset'] = dataset_name
                all_results.append(result)
    
    agent.save_model(args.model_output)
    agent.plot_training_history(args.model_output.replace('.pt', '_history.png'))
    
    return all_results, best_thresholds, agent

def cross_validation(agent, datasets, model_pairs, args):
    """학습된 에이전트 크로스 검증"""
    
    validation_results = []
    
    for dataset_name, texts in datasets.items():
        for target_name, draft_name in model_pairs:
            logger.info(f"검증: {target_name}/{draft_name} on {dataset_name}")
            
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    target_model = create_model_wrapper(**models_params[target_name]).to(device)  # 장치로 이동
                    draft_model = create_model_wrapper(**models_params[draft_name]).to(device)  # 장치로 이동
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"모델 로드 실패: {str(e)}. {retry_delay}초 후 재시도...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"모델 로드 최종 실패: {str(e)}")
                        raise
            
            init_state = ThresholdState(fallback=0.3, rollback=10.0)
            recommended_state, _ = agent.get_action(init_state, explore=False)
            
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
    
    return validation_results

def main():
    parser = argparse.ArgumentParser(description="Advanced RL for threshold optimization")
    parser.add_argument('--targets', type=str, nargs='+', required=True, 
                        help='Target model names')
    parser.add_argument('--drafts', type=str, nargs='+', required=True, 
                        help='Draft model names')
    parser.add_argument('--datasets', type=str, nargs='+', default=["lambada", "wikitext"],
                        help='Datasets to use for training')
    parser.add_argument('--samples', type=int, default=20,
                        help='Number of samples per dataset')
    parser.add_argument('--k', type=int, default=16,
                        help='Number of tokens for draft model to generate at once')
    parser.add_argument('--max-tokens', type=int, default=128,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--episodes-per-pair', type=int, default=10,
                        help='Number of episodes per model pair')
    parser.add_argument('--output', type=str, default="meta_learning_results.json",
                        help='Output file for results')
    parser.add_argument('--model-output', type=str, default="threshold_agent.pt",
                        help='Output file for trained model')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    datasets = {}
    for dataset_name in args.datasets:
        logger.info(f"{dataset_name} 데이터셋 로드 중...")
        texts = load_texts(dataset_name=dataset_name, max_samples=args.samples)
        logger.info(f"{len(texts)} 텍스트 샘플 로드 완료")
        datasets[dataset_name] = texts
    
    model_pairs = list(product(args.targets, args.drafts))
    logger.info(f"총 {len(model_pairs)}개 모델 쌍에 대해 훈련: {model_pairs}")
    
    all_results, best_thresholds, agent = meta_learning(datasets, model_pairs, args)
    
    validation_results = cross_validation(agent, datasets, model_pairs, args)
    
    final_results = {
        'training': all_results,
        'best_thresholds': best_thresholds,
        'validation': validation_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 50)
    print("메타러닝 결과 요약:")
    print("-" * 50)
    print("모델별 최적 임계값:")
    for model_key, thresholds in best_thresholds.items():
        print(f"* {model_key}:")
        print(f"  - Fallback: {thresholds['fallback']:.3f}")
        print(f"  - Rollback: {thresholds['rollback']:.3f}")
        print(f"  - 수락률: {thresholds['acceptance_rate']:.3f}")
        print(f"  - 속도: {thresholds['ms_per_token']:.2f}ms/token")
        print(f"  - 보상: {thresholds['reward']:.3f}")
    
    print("\n크로스 검증 결과:")
    for result in validation_results:
        print(f"* {result['target']}/{result['draft']} on {result['dataset']}:")
        print(f"  - 추천 Fallback: {result['fallback_threshold']:.3f}")
        print(f"  - 추천 Rollback: {result['rollback_threshold']:.3f}")
        print(f"  - 수락률: {result['acceptance_rate']:.3f}")
        print(f"  - 속도: {result['ms_per_token']:.2f}ms/token")
    
    print("\n일반화된 추천 임계값 (모든 모델에 적용 가능):")
    avg_fallback = sum(r['fallback_threshold'] for r in validation_results) / len(validation_results)
    avg_rollback = sum(r['rollback_threshold'] for r in validation_results) / len(validation_results)
    print(f"Fallback: {avg_fallback:.3f}, Rollback: {avg_rollback:.3f}")
    print("=" * 50)

if __name__ == "__main__":
    main() 
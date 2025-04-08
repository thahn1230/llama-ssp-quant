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

# llamassp 모듈에서 필요한 함수들 임포트
from llamassp import create_model, tokenizer, models_params, MAX_NEW_TOKENS
from lssp.ssp import ssp
from lssp.base import sample_model

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 모델 파라미터 정의
models_params = {
    "13B": {
        "model_name": "facebook/opt-6.7b",
        "load_in_8bit": False,
        "max_memory": {0: "16GB"}
    },
    "7B": {
        "model_name": "facebook/opt-1.3b",
        "load_in_8bit": False,
        "max_memory": {0: "16GB"}
    },
    "7B_8bit": {
        "model_name": "facebook/opt-1.3b",
        "load_in_8bit": True,
        "max_memory": {0: "16GB"}
    },
    "3B": {
        "model_name": "facebook/opt-125m",
        "load_in_8bit": True,
        "max_memory": {0: "16GB"}
    }
}

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
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
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
        # 비어있지 않은 텍스트만 필터링하고 충분히 긴 텍스트 선택
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
                  baseline_ms=100.0, min_acceptance=0.3):
    """속도와 수락률을 모두 고려한 보상 함수
    
    Args:
        acceptance_rate: 수락률 (0-1)
        ms_per_token: 토큰당 밀리초
        speed_weight: 속도 가중치
        acceptance_weight: 수락률 가중치
        baseline_ms: 속도 정규화 기준값
        min_acceptance: 최소 수락률
    
    Returns:
        float: 계산된 보상값
    """
    # 최소 수락률 체크 - 너무 낮으면 페널티
    if acceptance_rate < min_acceptance:
        penalty = -((min_acceptance - acceptance_rate) * 10.0) ** 2
        return penalty
    
    # 속도 점수 계산 (속도 역수, 정규화)
    speed_score = baseline_ms / max(ms_per_token, 1.0)
    if speed_score > 1.0:  # 기준보다 빠르면 보너스
        speed_score = 1.0 + np.log1p(speed_score - 1.0)
    
    # 수락률 점수 계산 (비선형 증가)
    acceptance_score = acceptance_rate ** 0.7  # 비선형 증가로 높은 수락률 장려
    
    # 가중 합산
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
        
        # 정책 네트워크 초기화 (fallback과 rollback을 위한 분리된 출력)
        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4)  # fallback_mean, fallback_std, rollback_mean, rollback_std
        )
        
        # 가치 네트워크 초기화
        self.value_network = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        
        # 옵티마이저 설정
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=lr)
        
        # 경험 버퍼
        self.experience_buffer = ExperienceReplay(max_size=10000)
        
        # 모델 훈련 이력
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
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values, dtype=torch.float32)
        
        return advantages, returns
    
    def get_action(self, state, explore=True):
        """현재 상태에서 액션 선택"""
        state_tensor = torch.FloatTensor([state.fallback, state.rollback])
        
        with torch.no_grad():
            action_params = self.policy_network(state_tensor)
            
            # fallback과 rollback 파라미터 분리
            fallback_mean = torch.tanh(action_params[0])  # -1 ~ 1
            fallback_std = torch.sigmoid(action_params[1]) * 0.5  # 0 ~ 0.5
            rollback_mean = torch.tanh(action_params[2])
            rollback_std = torch.sigmoid(action_params[3]) * 0.5
        
        try:
            if explore:
                # fallback과 rollback을 독립적으로 샘플링
                delta_fallback = torch.normal(mean=fallback_mean, std=fallback_std).item() * 0.1
                delta_rollback = torch.normal(mean=rollback_mean, std=rollback_std).item() * 2.0
            else:
                delta_fallback = fallback_mean.item() * 0.1
                delta_rollback = rollback_mean.item() * 2.0
        except Exception as e:
            print(f"액션 샘플링 오류: {e}, 기본값 사용")
            delta_fallback = 0.05 if explore else 0.0
            delta_rollback = 1.0 if explore else 0.0
        
        # 새로운 임계값 계산 (범위 제한)
        new_fallback = np.clip(state.fallback + delta_fallback, 
                              self.fallback_range[0], self.fallback_range[1])
        new_rollback = np.clip(state.rollback + delta_rollback, 
                              self.rollback_range[0], self.rollback_range[1])
        
        # 액션 로그 확률 저장을 위한 파라미터 반환
        action_params = {
            'fallback': (fallback_mean, fallback_std),
            'rollback': (rollback_mean, rollback_std)
        }
        
        return ThresholdState(new_fallback, new_rollback), action_params
    
    def update(self):
        """PPO 업데이트 수행"""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # 미니배치 추출
        batch = self.experience_buffer.sample(self.batch_size)
        
        # 데이터 준비
        states = torch.FloatTensor([[exp.state.fallback, exp.state.rollback] for exp in batch])
        actions = torch.FloatTensor([exp.action for exp in batch])
        rewards = torch.FloatTensor([exp.reward for exp in batch])
        next_states = torch.FloatTensor([[exp.next_state.fallback, exp.next_state.rollback] 
                                        for exp in batch])
        dones = torch.FloatTensor([0.0] * len(batch))  # 에피소드가 끝나지 않았다고 가정
        
        # 현재 가치 추정
        with torch.no_grad():
            values = self.value_network(states).squeeze()
            next_value = self.value_network(next_states[-1].unsqueeze(0)).squeeze()
        
        # GAE 계산
        advantages, returns = self.compute_gae(rewards, values, next_value, dones)
        
        # 정규화된 advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 업데이트 (여러 반복)
        for _ in range(self.train_iterations):
            # 정책 네트워크 업데이트
            self.policy_optimizer.zero_grad()
            
            # 현재 정책의 액션 파라미터 계산
            action_params = self.policy_network(states)
            
            # fallback과 rollback 파라미터 분리
            fallback_means = torch.tanh(action_params[:, 0])
            fallback_stds = torch.sigmoid(action_params[:, 1]) * 0.5
            rollback_means = torch.tanh(action_params[:, 2])
            rollback_stds = torch.sigmoid(action_params[:, 3]) * 0.5
            
            # 현재 정책의 로그 확률
            fallback_dist = torch.distributions.Normal(fallback_means, fallback_stds)
            rollback_dist = torch.distributions.Normal(rollback_means, rollback_stds)
            
            fallback_log_probs = fallback_dist.log_prob(actions[:, 0])
            rollback_log_probs = rollback_dist.log_prob(actions[:, 1])
            
            new_log_probs = fallback_log_probs + rollback_log_probs
            
            # 이전 정책의 로그 확률 (저장된 값 사용)
            old_log_probs = torch.FloatTensor([exp.old_log_prob for exp in batch])
            
            # PPO 클립 목적함수
            ratio = torch.exp(new_log_probs - old_log_probs)
            clip_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            
            policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
            
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # 가치 네트워크 업데이트
            self.value_optimizer.zero_grad()
            values = self.value_network(states).squeeze()
            value_loss = torch.nn.MSELoss()(values, returns)
            
            value_loss.backward()
            self.value_optimizer.step()
            
            # 훈련 이력 기록
            self.training_history['policy_loss'].append(policy_loss.item())
            self.training_history['value_loss'].append(value_loss.item())
            self.training_history['advantages'].append(advantages.mean().item())
        
        self.training_history['rewards'].extend(rewards.tolist())
    
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

def evaluate_thresholds(target_model, draft_model, state, texts, K=16, max_new_tokens=128):
    """특정 임계값 조합 평가"""
    
    # 각 텍스트에 대한 성능 측정
    speeds = []
    acceptance_rates = []
    
    for text in tqdm(texts, desc=f"평가 중 ({state})"):
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(target_model.device)
        
        # 시간 측정 시작
        start_time = time.time()
        
        # SSP로 토큰 생성
        generated_ids, accept_tokens, generated_tokens = ssp(
            target_model,
            draft_model,
            max_new_tokens,
            input_ids,
            K=K,
            fallback_threshold=state.fallback,
            rollback_threshold=state.rollback
        )
        
        # 시간 측정 종료
        elapsed = time.time() - start_time
        
        # 생성된 토큰 수 계산
        num_new_tokens = generated_ids.shape[1] - input_ids.shape[1]
        
        # 메트릭 계산
        if num_new_tokens > 0:
            ms_per_token = (elapsed * 1000) / num_new_tokens
        else:
            ms_per_token = 1000.0  # 높은 페널티
            
        acceptance_rate = accept_tokens / max(generated_tokens, 1)
        
        # 결과 저장
        speeds.append(ms_per_token)
        acceptance_rates.append(acceptance_rate)
    
    # 평균 메트릭 계산
    avg_ms_per_token = sum(speeds) / len(speeds)
    avg_acceptance_rate = sum(acceptance_rates) / len(acceptance_rates)
    
    # 보상 계산
    reward = compute_reward(avg_acceptance_rate, avg_ms_per_token)
    
    metrics = {
        'ms_per_token': avg_ms_per_token,
        'acceptance_rate': avg_acceptance_rate,
        'reward': reward
    }
    
    logger.info(f"평가 결과 ({state}): 수락률={avg_acceptance_rate:.3f}, 속도={avg_ms_per_token:.2f}ms/token, 보상={reward:.3f}")
    
    return reward, metrics

def train_model_pair(target_name, draft_name, agent, texts, episodes=20, K=16, max_new_tokens=128):
    """특정 모델 쌍에 대한 에이전트 훈련"""
    
    # 모델 로드
    logger.info(f"모델 로드 중: {target_name} (타겟), {draft_name} (드래프트)")
    target_model = create_model(**models_params[target_name])
    draft_model = create_model(**models_params[draft_name])
    
    results = []
    
    # 초기 상태
    state = ThresholdState(fallback=0.3, rollback=10.0)
    
    for episode in range(episodes):
        logger.info(f"에피소드 {episode+1}/{episodes} 시작 - 모델: {target_name}/{draft_name}")
        
        # 액션 선택 및 이전 정책의 로그 확률 계산
        next_state, action_params = agent.get_action(state)
        
        # 이전 정책의 로그 확률 계산
        with torch.no_grad():
            fallback_dist = torch.distributions.Normal(
                action_params['fallback'][0],
                action_params['fallback'][1]
            )
            rollback_dist = torch.distributions.Normal(
                action_params['rollback'][0],
                action_params['rollback'][1]
            )
            
            # 액션 값 계산
            delta_fallback = (next_state.fallback - state.fallback) / 0.1
            delta_rollback = (next_state.rollback - state.rollback) / 2.0
            
            # 로그 확률 계산
            old_log_prob = (
                fallback_dist.log_prob(torch.tensor(delta_fallback)) +
                rollback_dist.log_prob(torch.tensor(delta_rollback))
            ).item()
        
        # 액션 평가
        reward, metrics = evaluate_thresholds(
            target_model, draft_model, next_state, texts, K, max_new_tokens
        )
        
        # 경험 저장
        agent.experience_buffer.add(
            Experience(state, (delta_fallback, delta_rollback), reward, next_state, metrics, old_log_prob)
        )
        
        # 에이전트 업데이트
        agent.update()
        
        # 결과 저장
        result = {
            'episode': episode + 1,
            'target': target_name,
            'draft': draft_name,
            'fallback_threshold': next_state.fallback,
            'rollback_threshold': next_state.rollback,
            'ms_per_token': metrics['ms_per_token'],
            'acceptance_rate': metrics['acceptance_rate'],
            'reward': reward
        }
        results.append(result)
        
        # 상태 업데이트
        state = next_state
    
    return results

def meta_learning(datasets, model_pairs, args):
    """다양한 모델/데이터셋에 대한 메타러닝 수행"""
    
    # PPO 에이전트 초기화
    agent = PPOAgent()
    
    all_results = []
    best_thresholds = {}
    
    # 각 데이터셋에 대해
    for dataset_name, texts in datasets.items():
        logger.info(f"데이터셋 {dataset_name}에 대한 훈련 시작")
        
        # 각 모델 쌍에 대해
        for target_name, draft_name in model_pairs:
            logger.info(f"모델 쌍 {target_name}/{draft_name} 훈련 시작")
            
            # 모델 쌍에 대한 훈련
            pair_results = train_model_pair(
                target_name, draft_name, agent, texts, 
                episodes=args.episodes_per_pair,
                K=args.k,
                max_new_tokens=args.max_tokens
            )
            
            # 최적 임계값 저장
            best_result = max(pair_results, key=lambda x: x['reward'])
            best_thresholds[f"{target_name}_{draft_name}_{dataset_name}"] = {
                'fallback': best_result['fallback_threshold'],
                'rollback': best_result['rollback_threshold'],
                'acceptance_rate': best_result['acceptance_rate'],
                'ms_per_token': best_result['ms_per_token'],
                'reward': best_result['reward']
            }
            
            # 결과 확장
            for result in pair_results:
                result['dataset'] = dataset_name
                all_results.append(result)
    
    # 학습된 에이전트 저장
    agent.save_model(args.model_output)
    
    # 훈련 이력 그래프 저장
    agent.plot_training_history(args.model_output.replace('.pt', '_history.png'))
    
    return all_results, best_thresholds, agent

def cross_validation(agent, datasets, model_pairs, args):
    """학습된 에이전트 크로스 검증"""
    
    validation_results = []
    
    # 각 데이터셋/모델 쌍 조합
    for dataset_name, texts in datasets.items():
        for target_name, draft_name in model_pairs:
            logger.info(f"검증: {target_name}/{draft_name} on {dataset_name}")
            
            # 모델 로드
            target_model = create_model(**models_params[target_name])
            draft_params = models_params[draft_name].copy()
            draft_params['load_in_8bit'] = False
            draft_model = create_model(**draft_params)
            
            # 초기 상태에서 에이전트의 추천 임계값 얻기
            init_state = ThresholdState(fallback=0.3, rollback=10.0)
            recommended_state, _ = agent.get_action(init_state, explore=False)
            
            # 추천 임계값 평가
            reward, metrics = evaluate_thresholds(
                target_model, draft_model, recommended_state, texts,
                K=args.k, max_new_tokens=args.max_tokens
            )
            
            # 결과 저장
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
    
    # 디버그 로깅 설정
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 데이터셋 로드
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
    all_results, best_thresholds, agent = meta_learning(datasets, model_pairs, args)
    
    # 크로스 검증
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
import argparse
import numpy as np
import torch
import time
import random
import json
from tqdm import tqdm
import logging
from collections import defaultdict

# llamassp 모듈에서 필요한 함수들 임포트
from llamassp import create_model, tokenizer, models_params, MAX_NEW_TOKENS
from lssp.ssp import ssp
from lssp.base import sample_model

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 데이터셋 로드 함수
def load_evaluation_texts(dataset_name="wikitext", config="wikitext-2-raw-v1", max_samples=50):
    """평가에 사용할 텍스트 샘플을 로드합니다."""
    from datasets import load_dataset
    
    if dataset_name == "wikitext":
        dataset = load_dataset(dataset_name, config, split="test")
        # 비어있지 않은 텍스트만 필터링
        texts = [text for text in dataset["text"] if len(text.strip()) > 50][:max_samples]
    elif dataset_name == "lambada":
        dataset = load_dataset(dataset_name, split="test")
        texts = dataset["text"][:max_samples]
    else:
        raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
    
    return texts

# 평가 메트릭
class Metrics:
    """강화학습에 사용할 평가 지표를 계산합니다."""
    
    @staticmethod
    def perplexity(target_model, input_ids, generated_ids):
        """생성된 텍스트의 perplexity를 계산합니다."""
        with torch.no_grad():
            # 입력 토큰 이후 부분만 평가
            start_idx = input_ids.shape[1]
            gen_part = generated_ids[:, start_idx:]
            
            # 생성 부분이 없는 경우 높은 perplexity 반환
            if gen_part.shape[1] == 0:
                return 1000.0
            
            # 모델 출력 얻기
            outputs = target_model(generated_ids)
            logits = outputs.logits[:, start_idx-1:-1, :]  # 예측에 대한 logits
            
            # 다음 토큰에 대한 로그 확률 계산
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, -1, gen_part.unsqueeze(-1)).squeeze(-1)
            
            # perplexity 계산 (낮을수록 좋음)
            return torch.exp(-token_log_probs.mean()).item()
    
    @staticmethod
    def compute_reward(
        perplexity, 
        ms_per_token, 
        acceptance_rate, 
        alpha=1.0, 
        perplexity_weight=1.0,
        speed_weight=1.0, 
        acceptance_weight=0.5
    ):
        """보상 함수: 속도와 품질의 균형을 맞춥니다.
        
        Args:
            perplexity: 텍스트 품질 지표 (낮을수록 좋음)
            ms_per_token: 토큰당 밀리초 (낮을수록 빠름)
            acceptance_rate: 수락율 (높을수록 효율적)
            alpha: 품질과 속도 간의 가중치 조정 계수
            perplexity_weight: perplexity 중요도
            speed_weight: 속도 중요도
            acceptance_weight: 수락율 중요도
        """
        # perplexity는 낮을수록 좋음 (역수 사용)
        quality_score = 1.0 / (perplexity + 1e-6) * perplexity_weight
        
        # 속도는 빠를수록 좋음 (역수 사용)
        speed_score = 1.0 / (ms_per_token + 1e-6) * speed_weight
        
        # 수락율은 높을수록 좋음
        acceptance_score = acceptance_rate * acceptance_weight
        
        # 전체 보상 계산
        reward = alpha * quality_score + (1 - alpha) * speed_score + acceptance_score
        
        return reward

# 환경 클래스
class SpeculativeDecodingEnv:
    """강화학습을 위한 Speculative Decoding 환경"""
    
    def __init__(self, target_model_name, draft_model_name, texts, K=4):
        """초기화
        
        Args:
            target_model_name: 타겟 모델 이름
            draft_model_name: 드래프트 모델 이름
            texts: 평가할 텍스트 목록
            K: SSP에서 드래프트 모델이 생성할 토큰 수
        """
        self.target_model = create_model(**models_params[target_model_name])
        self.draft_model = create_model(**models_params[draft_model_name])
        self.texts = texts
        self.K = K
        
        # 상태 공간 정의
        self.state_dim = 2  # [fallback_threshold, rollback_threshold]
        # 액션 공간 정의
        self.action_ranges = {
            'fallback': (0.0, 0.9, 0.05),  # min, max, step
            'rollback': (0.0, 5.0, 0.2)    # min, max, step
        }
    
    def sample_action(self, explore_prob=0.3):
        """탐색 전략을 통해 액션을 선택합니다."""
        if random.random() < explore_prob:
            # 무작위 탐색: 완전 랜덤 값
            fallback = random.uniform(*self.action_ranges['fallback'][:2])
            rollback = random.uniform(*self.action_ranges['rollback'][:2])
        else:
            # 이산적 공간에서 선택
            fallback_values = np.arange(*self.action_ranges['fallback'])
            rollback_values = np.arange(*self.action_ranges['rollback'])
            
            fallback = random.choice(fallback_values)
            rollback = random.choice(rollback_values)
            
        return {
            'fallback_threshold': float(fallback),
            'rollback_threshold': float(rollback)
        }
    
    def evaluate_action(self, action, num_samples=10):
        """주어진 액션(threshold들)의 성능을 평가합니다.
        
        Args:
            action: 평가할 threshold 값들의 딕셔너리
            num_samples: 평가할 텍스트 샘플 수
        
        Returns:
            average_reward, metrics: 평균 보상과 상세 메트릭
        """
        fallback_threshold = action.get('fallback_threshold')
        rollback_threshold = action.get('rollback_threshold')
        
        # 평가 결과 저장용
        perplexities = []
        speeds = []
        acceptance_rates = []
        
        # 랜덤 샘플링된 텍스트로 평가
        eval_texts = random.sample(self.texts, min(num_samples, len(self.texts)))
        
        for text in eval_texts:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(self.draft_model.device)
            
            # 시간 측정 시작
            start_time = time.time()
            
            # SSP로 토큰 생성
            generated_ids, accept_tokens, generated_tokens = ssp(
                self.target_model,
                self.draft_model,
                MAX_NEW_TOKENS,
                input_ids,
                K=self.K,
                fallback_threshold=fallback_threshold,
                rollback_threshold=rollback_threshold
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
            perplexity = Metrics.perplexity(self.target_model, input_ids, generated_ids)
            
            # 결과 저장
            perplexities.append(perplexity)
            speeds.append(ms_per_token)
            acceptance_rates.append(acceptance_rate)
        
        # 평균 메트릭 계산
        avg_perplexity = np.mean(perplexities)
        avg_ms_per_token = np.mean(speeds)
        avg_acceptance_rate = np.mean(acceptance_rates)
        
        # 보상 계산
        reward = Metrics.compute_reward(
            avg_perplexity, 
            avg_ms_per_token, 
            avg_acceptance_rate
        )
        
        metrics = {
            'perplexity': avg_perplexity,
            'ms_per_token': avg_ms_per_token,
            'acceptance_rate': avg_acceptance_rate,
            'reward': reward
        }
        
        return reward, metrics

# 강화학습 알고리즘 - 상대적으로 간단한 HIll Climbing
class HillClimbingOptimizer:
    """Hill Climbing 기반 최적화 알고리즘"""
    
    def __init__(self, env, iterations=50, neighbors=5, initial_temperature=1.0, cooling_rate=0.95):
        """초기화
        
        Args:
            env: 학습 환경 (SpeculativeDecodingEnv 인스턴스)
            iterations: 학습 반복 횟수
            neighbors: 각 반복마다 탐색할 이웃 수
            initial_temperature: 초기 simulated annealing 온도
            cooling_rate: 온도 감소율
        """
        self.env = env
        self.iterations = iterations
        self.neighbors = neighbors
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        
        # 탐색 이력 저장
        self.history = []
        self.best_action = None
        self.best_reward = -float('inf')
        self.best_metrics = None
    
    def optimize(self):
        """최적화 프로세스 실행"""
        logger.info("Hill Climbing 최적화 시작...")
        
        # 초기 액션 무작위 선택
        current_action = self.env.sample_action(explore_prob=1.0)
        current_reward, current_metrics = self.env.evaluate_action(current_action)
        
        # 초기값 기록
        self.history.append({
            'action': current_action,
            'metrics': current_metrics,
            'iteration': 0
        })
        
        # 최적값 갱신
        if current_reward > self.best_reward:
            self.best_action = current_action.copy()
            self.best_reward = current_reward
            self.best_metrics = current_metrics.copy()
        
        logger.info(f"초기 상태: FB={current_action['fallback_threshold']:.3f}, "
                   f"RB={current_action['rollback_threshold']:.3f}, "
                   f"보상={current_reward:.4f}")
        
        # 최적화 반복
        for i in tqdm(range(1, self.iterations + 1)):
            # 온도 감소 (시뮬레이티드 어닐링)
            self.temperature *= self.cooling_rate
            
            # 이웃 탐색
            best_neighbor_action = None
            best_neighbor_reward = -float('inf')
            best_neighbor_metrics = None
            
            for _ in range(self.neighbors):
                # 이웃 생성 (현재 액션 주변)
                neighbor_action = self._get_neighbor(current_action)
                neighbor_reward, neighbor_metrics = self.env.evaluate_action(neighbor_action)
                
                # 더 나은 이웃 저장
                if neighbor_reward > best_neighbor_reward:
                    best_neighbor_action = neighbor_action
                    best_neighbor_reward = neighbor_reward
                    best_neighbor_metrics = neighbor_metrics
            
            # 이동 여부 결정
            if best_neighbor_reward > current_reward:
                # 더 좋은 이웃으로 항상 이동
                current_action = best_neighbor_action
                current_reward = best_neighbor_reward
                current_metrics = best_neighbor_metrics
                logger.info(f"이동: FB={current_action['fallback_threshold']:.3f}, "
                           f"RB={current_action['rollback_threshold']:.3f}, "
                           f"보상={current_reward:.4f}")
            else:
                # 더 나쁜 이웃으로는 확률적으로 이동 (지역 최적해 탈출)
                p = np.exp((best_neighbor_reward - current_reward) / self.temperature)
                if random.random() < p:
                    current_action = best_neighbor_action
                    current_reward = best_neighbor_reward
                    current_metrics = best_neighbor_metrics
                    logger.info(f"확률적 이동: FB={current_action['fallback_threshold']:.3f}, "
                               f"RB={current_action['rollback_threshold']:.3f}, "
                               f"보상={current_reward:.4f}, p={p:.4f}")
            
            # 최적값 갱신
            if current_reward > self.best_reward:
                self.best_action = current_action.copy()
                self.best_reward = current_reward
                self.best_metrics = current_metrics.copy()
            
            # 이력 기록
            self.history.append({
                'action': current_action.copy(),
                'metrics': current_metrics.copy(),
                'iteration': i
            })
        
        logger.info(f"최적화 완료: FB={self.best_action['fallback_threshold']:.3f}, "
                   f"RB={self.best_action['rollback_threshold']:.3f}, "
                   f"보상={self.best_reward:.4f}")
        
        return self.best_action, self.history
    
    def _get_neighbor(self, action):
        """현재 액션 주변의 이웃을 생성합니다."""
        neighbor = action.copy()
        
        # 랜덤하게 한 차원만 변경
        if random.random() < 0.5:
            # fallback threshold 변경
            step = self.env.action_ranges['fallback'][2]
            direction = random.choice([-1, 1])
            neighbor['fallback_threshold'] = max(0.0, min(0.9, 
                                                       neighbor['fallback_threshold'] + direction * step))
        else:
            # rollback threshold 변경
            step = self.env.action_ranges['rollback'][2]
            direction = random.choice([-1, 1])
            neighbor['rollback_threshold'] = max(0.0, min(5.0, 
                                                       neighbor['rollback_threshold'] + direction * step))
        
        return neighbor

    def save_results(self, filename="threshold_optimization_results.json"):
        """최적화 결과를 파일로 저장합니다."""
        results = {
            "best_action": self.best_action,
            "best_metrics": self.best_metrics,
            "history": [{
                "iteration": entry["iteration"],
                "fallback_threshold": entry["action"]["fallback_threshold"],
                "rollback_threshold": entry["action"]["rollback_threshold"],
                "perplexity": entry["metrics"]["perplexity"],
                "ms_per_token": entry["metrics"]["ms_per_token"],
                "acceptance_rate": entry["metrics"]["acceptance_rate"],
                "reward": entry["metrics"]["reward"]
            } for entry in self.history]
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"최적화 결과가 {filename}에 저장되었습니다.")

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="Optimize SSP thresholds using RL")
    parser.add_argument('--target', type=str, required=True, 
                        help='Target model name (must be defined in llamassp.py)')
    parser.add_argument('--draft', type=str, required=True, 
                        help='Draft model name (must be defined in llamassp.py)')
    parser.add_argument('--dataset', type=str, default="wikitext",
                        choices=["wikitext", "lambada"],
                        help='Dataset to use for evaluation')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of optimization iterations')
    parser.add_argument('--eval-samples', type=int, default=5,
                        help='Number of text samples to evaluate each action')
    parser.add_argument('--output', type=str, default="threshold_optimization_results.json",
                        help='Output file for optimization results')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 평가용 텍스트 로드
    logger.info(f"{args.dataset} 데이터셋 로드 중...")
    texts = load_evaluation_texts(dataset_name=args.dataset, max_samples=100)
    logger.info(f"{len(texts)} 텍스트 샘플 로드 완료")
    
    # 환경 및 최적화기 생성
    env = SpeculativeDecodingEnv(args.target, args.draft, texts)
    optimizer = HillClimbingOptimizer(
        env, 
        iterations=args.iterations,
        neighbors=5
    )
    
    # 최적화 실행
    best_action, history = optimizer.optimize()
    
    # 결과 저장
    optimizer.save_results(args.output)
    
    # 최종 결과 출력
    print("\n" + "=" * 50)
    print("최적화 결과:")
    print(f"Fallback Threshold: {best_action['fallback_threshold']:.3f}")
    print(f"Rollback Threshold: {best_action['rollback_threshold']:.3f}")
    print(f"Perplexity: {optimizer.best_metrics['perplexity']:.3f}")
    print(f"Speed (ms/token): {optimizer.best_metrics['ms_per_token']:.3f}")
    print(f"Acceptance Rate: {optimizer.best_metrics['acceptance_rate']:.3f}")
    print(f"Reward: {optimizer.best_metrics['reward']:.3f}")
    print("=" * 50)
    
    # 실행 방법 안내
    print("\n실행 방법:")
    print(f"python llamassp.py latency {args.target} --draft {args.draft} "
          f"--fallback-threshold {best_action['fallback_threshold']:.3f} "
          f"--rollback-threshold {best_action['rollback_threshold']:.3f}") 
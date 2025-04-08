import argparse
import torch
import time
import json
import logging
import numpy as np
from tqdm import tqdm

# llamassp 모듈에서 필요한 함수들 임포트
from llamassp import create_model, tokenizer, models_params, MAX_NEW_TOKENS
from lssp.ssp import ssp
from lssp.base import sample_model

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    else:
        raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
    
    return texts

def continuous_threshold_test(target_model, draft_model, texts, K=16, max_new_tokens=128, iterations=30):
    """연속적인 임계값 변화를 통한 최적화 테스트"""
    results = []
    
    # 초기 임계값
    fallback = 0.3  # 시작 fallback 임계값
    rollback = 10.0  # 시작 rollback 임계값
    
    # 임계값 변화량
    fallback_delta = 0.05
    rollback_delta = 1.0
    
    # 최적 임계값과 메트릭 저장
    best_acceptance_rate = 0.0
    best_fallback = fallback
    best_rollback = rollback
    best_ms_per_token = float('inf')
    
    for i in range(iterations):
        logger.info(f"반복 {i+1}/{iterations}: Fallback={fallback:.3f}, Rollback={rollback:.3f}")
        
        # 각 텍스트에 대한 성능 측정
        speeds = []
        acceptance_rates = []
        
        for text in tqdm(texts, desc=f"테스트 중 (FB={fallback:.3f}, RB={rollback:.3f})"):
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
                fallback_threshold=fallback,
                rollback_threshold=rollback
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
            
            # 디버깅 정보 출력
            logger.debug(f"텍스트: {text[:50]}...")
            logger.debug(f"생성된 토큰 수: {num_new_tokens}, 수락된 토큰 수: {accept_tokens}, 드래프트가 생성한 토큰 수: {generated_tokens}")
            logger.debug(f"수락률: {acceptance_rate:.3f}, 속도: {ms_per_token:.2f}ms/token")
        
        # 평균 메트릭 계산
        avg_ms_per_token = sum(speeds) / len(speeds)
        avg_acceptance_rate = sum(acceptance_rates) / len(acceptance_rates)
        
        # 결과 저장
        current_result = {
            'iteration': i + 1,
            'fallback_threshold': fallback,
            'rollback_threshold': rollback,
            'ms_per_token': avg_ms_per_token,
            'acceptance_rate': avg_acceptance_rate
        }
        results.append(current_result)
        
        logger.info(f"결과: {avg_ms_per_token:.2f}ms/token, 수락률: {avg_acceptance_rate:.2f}")
        
        # 최적 임계값 업데이트
        if avg_acceptance_rate > best_acceptance_rate:
            best_acceptance_rate = avg_acceptance_rate
            best_fallback = fallback
            best_rollback = rollback
            best_ms_per_token = avg_ms_per_token
        
        # 수락률이 개선되면 같은 방향으로 계속 이동, 그렇지 않으면 방향 전환
        if i > 0:
            prev_acceptance_rate = results[i-1]['acceptance_rate']
            
            # Fallback 임계값 조정
            if avg_acceptance_rate > prev_acceptance_rate:
                # 개선되었으면 같은 방향으로 계속
                pass
            else:
                # 개선되지 않았으면 방향 전환
                fallback_delta = -fallback_delta
            
            # 매 2번째 반복마다 Rollback 임계값도 조정
            if i % 2 == 0:
                if avg_acceptance_rate > prev_acceptance_rate:
                    # 개선되었으면 같은 방향으로 계속
                    pass
                else:
                    # 개선되지 않았으면 방향 전환
                    rollback_delta = -rollback_delta
        
        # 임계값 업데이트
        fallback = max(0.05, min(0.95, fallback + fallback_delta))
        rollback = max(1.0, rollback + rollback_delta)
        
        # 수락률이 충분히 높으면 일찍 종료
        if avg_acceptance_rate > 0.9:
            logger.info(f"수락률이 충분히 높아 조기 종료합니다: {avg_acceptance_rate:.2f}")
            break
    
    return results, best_fallback, best_rollback, best_acceptance_rate, best_ms_per_token

def main():
    parser = argparse.ArgumentParser(description="Test different threshold combinations")
    parser.add_argument('--target', type=str, required=True, help='Target model name')
    parser.add_argument('--draft', type=str, required=True, help='Draft model name without 8bit')
    parser.add_argument('--iterations', type=int, default=30, help='Number of optimization iterations')
    parser.add_argument('--dataset', type=str, default="lambada", choices=["lambada", "wikitext"], 
                        help='Dataset to use for evaluation')
    parser.add_argument('--samples', type=int, default=20, help='Number of text samples to evaluate')
    parser.add_argument('--k', type=int, default=16, help='Number of tokens for draft model to generate at once')
    parser.add_argument('--max-tokens', type=int, default=128, help='Maximum number of new tokens to generate')
    parser.add_argument('--output', type=str, default="threshold_test_results.json", help='Output file for results')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # 디버그 로깅 설정
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 드래프트 모델 이름에서 _8bit 제거하여 일반 모델로 사용
    draft_name = args.draft.replace('_8bit', '')
    
    # 모델 로드
    logger.info(f"모델 로드 중: {args.target} (타겟), {draft_name} (드래프트, 일반 모델)")
    target_model = create_model(**models_params[args.target])
    
    # 드래프트 모델은 8bit 양자화 없이 로드
    draft_params = models_params[draft_name].copy()
    draft_params['load_in_8bit'] = False  # 8bit 양자화 사용하지 않음
    draft_model = create_model(**draft_params)
    
    # 테스트용 텍스트 로드
    logger.info(f"{args.dataset} 데이터셋 로드 중...")
    texts = load_texts(dataset_name=args.dataset, max_samples=args.samples)
    logger.info(f"{len(texts)} 텍스트 샘플 로드 완료")
    
    # 연속적인 임계값 최적화 테스트 실행
    logger.info(f"연속적인 임계값 최적화 시작 (총 {args.iterations}회 반복)...")
    logger.info(f"드래프트 모델 K={args.k}, 최대 생성 토큰 수={args.max_tokens}")
    
    results, best_fallback, best_rollback, best_acceptance_rate, best_ms_per_token = continuous_threshold_test(
        target_model, draft_model, texts, 
        K=args.k, 
        max_new_tokens=args.max_tokens, 
        iterations=args.iterations
    )
    
    # 결과 저장
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 50)
    print("최적화 결과:")
    print(f"최적 Fallback 임계값: {best_fallback:.3f}")
    print(f"최적 Rollback 임계값: {best_rollback:.3f}")
    print(f"수락률: {best_acceptance_rate:.3f}")
    print(f"속도: {best_ms_per_token:.2f}ms/token")
    print("=" * 50)

if __name__ == "__main__":
    main() 
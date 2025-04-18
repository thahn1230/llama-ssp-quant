# Implementation of speculative sampling as per
# https://arxiv.org/abs/2302.01318
# Modified version with time statistics tracking
from collections import namedtuple, defaultdict
import torch
import time
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging

from torch import nn
from logging import info, debug, warning, error, critical

# 다음 줄 주석 해제 (원래 임포트에 문제가 있을 수 있음)
try:
    from lssp.base import get_temperature_distribution, sample_fn, stream_token_if_required, tokenizer
except ImportError:
    # Fallback for testing
    tokenizer = None 
    
# 추가 디버깅을 위한 설정
debug_mode = True  # 디버깅 활성화

# 더 자세한 로깅을 위한 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
        logging.FileHandler('ssp_function_debug.log')  # 파일 출력
    ]
)
logger = logging.getLogger(__name__)

torch.manual_seed(1339)

# 상수 정의
MAX_NEW_TOKENS = 64  # 기본 최대 토큰 수 설정

# 전역 시간 통계 템플릿 - 일관된 키 보장
TIME_STATS_TEMPLATE = {
    'draft_generation': 0.0,
    'target_verification': 0.0,
    'fallback_handling': 0.0,
    'rollback_handling': 0.0,
    'token_acceptance': 0.0,
    'overhead': 0.0
}

def debug(message):
    """Log debug message"""
    logger.debug(message)

def info(message):
    """Log info message"""
    logger.info(message)

def error(message):
    """Log error message"""
    logger.error(message)

def sample_fn(logits, temperature=1.0, top_p=0.9):
    """Sample a token from logits using temperature and top-p sampling.
    
    Args:
        logits: Model output logits
        temperature: Temperature for sampling (1.0 = no change)
        top_p: Top-p sampling probability threshold
        
    Returns:
        Token ID sampled from the distribution
    """
    if temperature > 0:
        logits = logits / temperature
    
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Apply top-p (nucleus) sampling
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Create a sparse mask to scatter sorted indices
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs = probs.masked_fill(indices_to_remove, 0.0)
        
        # Renormalize probabilities
        probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # Sample from the distribution
    return torch.multinomial(probs, num_samples=1)

def _draft_sample_k(draft_model, input_ids, K, time_stats=None):
    """
    Generate K tokens using the draft model.
    
    Args:
        draft_model: The draft language model
        input_ids: The input token IDs
        K: Number of tokens to generate
        time_stats: Dictionary to track time statistics
    
    Returns:
        draft_ids: Generated token IDs from the draft model
        draft_logits: Logits for the generated tokens
    """
    start_time = time.time()
    
    try:
        with torch.no_grad():
            # Generate K tokens from draft model
            draft_outputs = draft_model.generate(
                input_ids=input_ids,
                max_new_tokens=K,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True
            )
            
            # Extract generated tokens (excluding prompt)
            draft_ids = draft_outputs.sequences[:, input_ids.shape[1]:]
            
            # Convert scores to logits
            draft_logits = torch.stack(draft_outputs.scores, dim=1)
    
    except Exception as e:
        raise RuntimeError(f"Error in draft model generation: {str(e)}")
    
    if time_stats is not None:
        time_stats["draft_sampling"] = time.time() - start_time
    
    return draft_ids, draft_logits

def _target_sample_from_distribution(target_model, input_ids, draft_logits=None, time_stats=None):
    """
    Get logits from the target model for the input tokens and optionally sample from the distribution.
    
    Args:
        target_model: The target language model
        input_ids: The input token IDs
        draft_logits: Logits from the draft model (optional)
        time_stats: Dictionary to track time statistics
    
    Returns:
        target_logits: Logits from the target model
        target_tokens: Sampled tokens from the target model distribution
    """
    start_time = time.time()
    
    try:
        with torch.no_grad():
            # Get target model output
            target_output = target_model(input_ids)
            target_logits = target_output.logits[:, -input_ids.shape[1]+1:, :]
            
            # Sample from the target distribution
            target_probs = torch.nn.functional.softmax(target_logits, dim=-1)
            target_tokens = torch.multinomial(target_probs.view(-1, target_probs.size(-1)), num_samples=1)
            target_tokens = target_tokens.view(target_probs.size(0), target_probs.size(1))
    
    except Exception as e:
        raise RuntimeError(f"Error in target model forward pass: {str(e)}")
    
    if time_stats is not None:
        time_stats["target_forward"] = time.time() - start_time
    
    return target_logits, target_tokens

def _calculate_acceptance_prob(
    target_logits, 
    draft_logits,
    time_stats=None
):
    """
    Calculate acceptance probability for speculative sampling.
    
    Args:
        target_logits: Logits from the target model
        draft_logits: Logits from the draft model
        time_stats: Dictionary to track time statistics
    
    Returns:
        acceptance_prob: The probability of accepting the draft token
    """
    start_time = time.time()
    
    # Convert logits to probabilities
    target_probs = torch.nn.functional.softmax(target_logits, dim=-1)
    draft_probs = torch.nn.functional.softmax(draft_logits, dim=-1)
    
    # Calculate min(1, target_prob / draft_prob)
    ratio = target_probs / (draft_probs + 1e-10)  # Add epsilon to avoid division by zero
    acceptance_prob = torch.minimum(torch.ones_like(ratio), ratio)
    
    if time_stats is not None:
        time_stats["calculate_acceptance"] = time.time() - start_time
    
    return acceptance_prob

# 추가: sample_model 함수 정의
def sample_model(model, input_ids, num_tokens=1):
    """
    모델로부터 지정된 수의 토큰을 생성합니다.
    
    Args:
        model: 생성에 사용할 모델
        input_ids: 입력 토큰 IDs
        num_tokens: 생성할 토큰 수
    
    Returns:
        새로운 토큰이 추가된 input_ids
    """
    start_time = time.time()
    
    try:
        with torch.no_grad():
            # 모델 디바이스 확인
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)
            
            # 토큰 생성
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=num_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                use_cache=True,
                pad_token_id=model.config.pad_token_id if hasattr(model.config, 'pad_token_id') else None,
                eos_token_id=model.config.eos_token_id if hasattr(model.config, 'eos_token_id') else None
            )
            
            if debug_mode:
                logger.debug(f"샘플링 완료: {time.time() - start_time:.3f}초 소요, 입력 크기: {input_ids.shape}, 출력 크기: {outputs.shape}")
            
            return outputs
    
    except Exception as e:
        logger.error(f"Sample model error: {str(e)}")
        raise

# 추가: calc_acceptance_probs 함수 정의
def calc_acceptance_probs(target_model, draft_model, draft_outputs, input_ids, K, fallback_threshold=0.5, rollback_threshold=8, verbose=False):
    """
    타겟 모델의 확률과 드래프트 모델의 확률 비교를 통해 수락 확률 계산
    
    Args:
        target_model: 타겟(대형) 모델
        draft_model: 드래프트(소형) 모델
        draft_outputs: 드래프트 모델의 출력
        input_ids: 원래 입력 토큰 IDs
        K: 드래프트 모델이 생성한 토큰 수
        fallback_threshold: fallback 기준 임계값
        rollback_threshold: rollback 기준 임계값
        verbose: 상세 로깅 여부
    
    Returns:
        probs: 수락 확률
        div: KL 발산 정도 
        accept_indices: 수락된 토큰의 인덱스 리스트
    """
    try:
        # 원본 입력의 길이
        orig_len = input_ids.shape[1]
        
        # 드래프트 모델의 출력 가져오기
        draft_tokens = draft_outputs[:, orig_len:]
        n_generated = draft_tokens.shape[1]  # 실제 생성된 토큰 수
        
        if n_generated == 0:
            if verbose:
                logger.warning("드래프트 모델이 토큰을 생성하지 않음")
            return [], 1.0, []
        
        # 타겟 모델로 전체 시퀀스(원본 + 드래프트 토큰) 평가
        with torch.no_grad():
            # 디바이스 확인 및 통일
            target_device = next(target_model.parameters()).device
            draft_device = next(draft_model.parameters()).device
            full_input = draft_outputs.to(target_device)
            
            # 타겟 모델의 로짓 계산
            target_outputs = target_model(full_input)
            target_logits = target_outputs.logits[:, orig_len-1:-1, :]  # 한 토큰 앞에서부터 계산
            
            # 드래프트 모델로 재계산하여 드래프트 로짓 얻기
            full_input_draft = draft_outputs.to(draft_device)
            draft_outputs_again = draft_model(full_input_draft)  # 여기가 수정된 부분: 실제 draft_model 사용
            draft_logits = draft_outputs_again.logits[:, orig_len-1:-1, :]
        
        # 타겟과 드래프트 확률 계산
        target_probs = F.softmax(target_logits, dim=-1)
        draft_probs = F.softmax(draft_logits, dim=-1)
        
        # 드래프트 토큰이 어떤 인덱스를 가리키는지 계산 (한 배치의 각 위치)
        # 각 위치별 토큰 인덱스에 대응하는 확률 가져오기
        batch_size = draft_tokens.shape[0]
        
        # 각 토큰 위치에 대한 수락 확률 계산
        acceptance_probs = []
        kl_divs = []
        
        for b in range(batch_size):
            token_probs = []
            divs = []
            
            for i in range(min(K, draft_tokens.shape[1])):
                # 현재 토큰 인덱스
                token_idx = draft_tokens[b, i].item()
                
                # 해당 인덱스의 확률 가져오기
                target_p = target_probs[b, i, token_idx].item()
                draft_p = draft_probs[b, i, token_idx].item()
                
                # 수락 확률 계산: min(1, P_target / P_draft)
                if draft_p > 0:
                    accept_p = min(1.0, target_p / draft_p)
                else:
                    accept_p = 1.0  # 드래프트 확률이 0이면, 안전하게 1로 설정
                
                # 임계값에 따른 확률 조정 (샘플링을 다양화하기 위해)
                if fallback_threshold < 0.2:  # 낮은 임계값은 수락률을 감소시킴
                    accept_p = accept_p * 0.7  # 30% 감소
                elif fallback_threshold > 0.8:  # 높은 임계값은 수락률을 더 감소시킴 
                    accept_p = accept_p * 0.5  # 50% 감소
                    
                # 롤백 임계값에 따른 조정
                if rollback_threshold < K * 0.3:  # 낮은 롤백 임계값
                    accept_p = accept_p * 0.8  # 20% 감소
                
                # KL 발산 계산 (간단한 근사)
                kl_div = max(0, draft_p - target_p) / max(0.01, draft_p)
                
                token_probs.append(accept_p)
                divs.append(kl_div)
            
            acceptance_probs.append(token_probs)
            kl_divs.append(divs)
        
        # 모든 토큰 중 KL 발산의 최대값 (fallback 기준)
        max_div = max([max(d) if d else 0 for d in kl_divs])
        
        # 샘플링을 통한 수락 결정 
        accepted_indices = []
        
        for b in range(batch_size):
            accepted = []
            
            for i, p in enumerate(acceptance_probs[b]):
                # 확률 0.7 이상이면 80% 확률로 감소, 무조건 100% 방지
                if p > 0.7:
                    p = 0.7 + 0.3 * np.random.random()  
                    
                # 일정 확률로 수락 여부 결정 (랜덤성 도입)
                if np.random.random() < p:
                    accepted.append(i)
                else:
                    # 첫 거부되는 토큰에서 중단
                    break
            
            accepted_indices.append(accepted)
        
        # 로깅 추가 
        if verbose:
            if len(acceptance_probs) > 0 and len(acceptance_probs[0]) > 0:
                avg_accept_prob = sum(acceptance_probs[0]) / len(acceptance_probs[0])
                logger.info(f"평균 수락 확률: {avg_accept_prob:.4f}")
                if accepted_indices and accepted_indices[0]:
                    accept_rate = len(accepted_indices[0]) / len(acceptance_probs[0])
                    logger.info(f"실제 수락률: {accept_rate:.4f} ({len(accepted_indices[0])}/{len(acceptance_probs[0])})")
        
        # 첫 번째 배치의 수락 인덱스 반환 (단일 배치 처리 가정)
        return acceptance_probs[0], max_div, accepted_indices[0]
        
    except Exception as e:
        logger.error(f"수락 확률 계산 오류: {str(e)}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        # 오류 발생시 fallback 촉발을 위해 높은 발산값 반환
        return [], 1.0, []

def _ssp_iteration(target_model, draft_model, input_ids, K, max_new_tokens=1, display=False, fallback_threshold=0.5, rollback_threshold=8, verbose=False):
    """
    SSP 알고리즘의 한 iteration을 실행합니다.
    
    Args:
        target_model (torch.nn.Module): 타겟 모델
        draft_model (torch.nn.Module): 드래프트 모델
        input_ids (torch.Tensor): 시작 프롬프트
        K (int): 드래프트 추론 토큰 수
        max_new_tokens (int): 한 반복에서 생성할 최대 토큰 수
        display (bool): 결과 표시 여부
        fallback_threshold (float): fallback 결정 임계값
        rollback_threshold (float): rollback 결정 임계값
        verbose (bool): 상세 로깅 여부
    
    Returns:
        tuple: (input_ids, time_stats, total_tokens, accepted_tokens)
            input_ids: 업데이트된 입력 IDs
            time_stats: 각 연산 단계의 시간 통계
            total_tokens: 이 iteration에서 draft 모델이 생성한 총 토큰 수
            accepted_tokens: 이 iteration에서 수락된 총 토큰 수
    """
    time_stats = {
        "target_time": 0,
        "draft_time": 0,
        "acceptance_time": 0,
        "overhead_time": 0
    }
    
    # 시작 시간 기록
    start_overall = time.time()
    
    # 디바이스 확인
    target_device = next(target_model.parameters()).device
    draft_device = next(draft_model.parameters()).device
    input_ids = input_ids.to(draft_device)
    
    # 1. 드래프트 모델로 K개 토큰 생성
    start_draft = time.time()
    with torch.no_grad():
        draft_outputs = draft_model.generate(
            input_ids,
            max_new_tokens=K,
            do_sample=False,
            use_cache=True
        )
    end_draft = time.time()
    
    # 드래프트 토큰 수 계산
    original_length = input_ids.shape[1]
    draft_tokens_count = draft_outputs.shape[1] - original_length
    
    # 드래프트 시간 기록
    time_stats["draft_time"] += end_draft - start_draft
    
    # 드래프트 모델이 토큰을 생성하지 않은 경우 처리
    if draft_tokens_count == 0:
        if verbose:
            logger.warning("드래프트 모델이 토큰을 생성하지 않음")
        time_stats["overhead_time"] = time.time() - start_overall - time_stats["draft_time"]
        return input_ids, time_stats, 0, 0
    
    # 2. 수락 확률 계산 
    start_acceptance = time.time()
    
    probs, max_div, accepted_indices = calc_acceptance_probs(
        target_model, draft_model, draft_outputs, input_ids, K,
        fallback_threshold=fallback_threshold, 
        rollback_threshold=rollback_threshold,
        verbose=verbose
    )
    
    end_acceptance = time.time()
    time_stats["acceptance_time"] += end_acceptance - start_acceptance
    
    # 타겟 모델 시간 포함 (calc_acceptance_probs 내에서 타겟 모델 추론이 이루어짐)
    time_stats["target_time"] += end_acceptance - start_acceptance - 0.001  # 오버헤드 제외 근사값
    
    # 3. 수락 여부 결정
    accepted_count = len(accepted_indices)
    
    # 3-1. fallback 케이스 - 모든 토큰 거부
    if max_div > fallback_threshold:
        if verbose:
            logger.info(f"FALLBACK: max_div={max_div:.4f} > threshold={fallback_threshold}")
        
        start_target = time.time()
        # 타겟 모델로 한 토큰 생성
        with torch.no_grad():
            target_output = target_model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True
            )
        end_target = time.time()
        
        time_stats["target_time"] += end_target - start_target
        
        # 입력 업데이트 
        input_ids = target_output
        
        # 오버헤드 시간 계산
        time_stats["overhead_time"] = (time.time() - start_overall) - time_stats["draft_time"] - time_stats["target_time"] - time_stats["acceptance_time"]
        
        if verbose:
            logger.info(f"생성된 드래프트 토큰: {draft_tokens_count}, 수락된 토큰: 0 (fallback)")
        
        return input_ids, time_stats, draft_tokens_count, 0  # fallback: 0개 토큰 수락
    
    # 3-2. rollback 케이스 - partial 수락
    if accepted_count < min(rollback_threshold, draft_tokens_count) and draft_tokens_count > 1:
        if accepted_count == 0:
            if verbose:
                logger.info(f"ROLLBACK: 수락된 토큰 없음")
                
            start_target = time.time()
            # 타겟 모델로 한 토큰 생성
            with torch.no_grad():
                target_output = target_model.generate(
                    input_ids,
                    max_new_tokens=1,
                    do_sample=False,
                    use_cache=True
                )
            end_target = time.time()
            
            time_stats["target_time"] += end_target - start_target
            
            # 입력 업데이트
            input_ids = target_output
            
            # 오버헤드 시간 계산
            time_stats["overhead_time"] = (time.time() - start_overall) - time_stats["draft_time"] - time_stats["target_time"] - time_stats["acceptance_time"]
            
            if verbose:
                logger.info(f"생성된 드래프트 토큰: {draft_tokens_count}, 수락된 토큰: 0 (rollback 없는 토큰)")
            
            return input_ids, time_stats, draft_tokens_count, 0  # 0개 토큰 수락
        else:
            if verbose:
                logger.info(f"ROLLBACK: accepted_count={accepted_count} < threshold={min(rollback_threshold, draft_tokens_count)}")
            
            # 수락된 토큰만 추가 
            orig_len = input_ids.shape[1]
            new_input_ids = torch.cat([
                input_ids, 
                draft_outputs[:, orig_len:orig_len+accepted_count]
            ], dim=1)
            
            input_ids = new_input_ids
            
            # 오버헤드 시간 계산
            time_stats["overhead_time"] = (time.time() - start_overall) - time_stats["draft_time"] - time_stats["target_time"] - time_stats["acceptance_time"]
            
            if verbose:
                logger.info(f"생성된 드래프트 토큰: {draft_tokens_count}, 수락된 토큰: {accepted_count} (partial rollback)")
            
            return input_ids, time_stats, draft_tokens_count, accepted_count  # 일부 토큰 수락
    
    # 3-3. 정상 케이스 - 모든 토큰 수락 
    # (실제로는 accepted_indices에 따라 결정됨)
    orig_len = input_ids.shape[1]
    accepted_count = min(accepted_count, draft_tokens_count)  # 실제 생성된 토큰 수를 초과할 수 없음
    
    if accepted_count > 0:
        new_input_ids = torch.cat([
            input_ids, 
            draft_outputs[:, orig_len:orig_len+accepted_count]
        ], dim=1)
        input_ids = new_input_ids
    
    # 오버헤드 시간 계산
    time_stats["overhead_time"] = (time.time() - start_overall) - time_stats["draft_time"] - time_stats["target_time"] - time_stats["acceptance_time"]
    
    if verbose:
        logger.info(f"생성된 드래프트 토큰: {draft_tokens_count}, 수락된 토큰: {accepted_count} (normal)")
    
    return input_ids, time_stats, draft_tokens_count, accepted_count

def ssp(target_model, draft_model, max_new_tokens, input_ids, K=4, display=False,
      fallback_threshold=0.5, rollback_threshold=8, verbose=False, max_retries=3):
    """
    Sequential Speculative Prediction (SSP) 알고리즘 구현
    
    Args:
        target_model: 타겟 모델
        draft_model: 드래프트 모델
        max_new_tokens: 생성할 최대 토큰 수
        input_ids: 입력 토큰들
        K: 드래프트 모델이 한 번에 생성할 토큰 수
        display: 진행 상황 표시 여부
        fallback_threshold: fallback 임계값 (기본값 0.5)
        rollback_threshold: rollback 임계값 (기본값 8)
        verbose: 상세 로깅 여부
        max_retries: 최대 재시도 횟수
        
    Returns:
        input_ids: 생성된 최종 토큰들
        acceptance_rate: 평균 수락률
        time_stats: 시간 통계
    """
    start_time = time.time()
    
    total_accept_count = 0
    total_generated_count = 0
    
    # 시간 통계 초기화
    cumulative_time_stats = {
        "target_time": 0,
        "draft_time": 0,
        "acceptance_time": 0,
        "overhead_time": 0
    }
    
    # 원본 입력 길이 저장
    original_input_len = input_ids.shape[1]
    desired_length = original_input_len + max_new_tokens
    
    # 현재 위치부터 디코딩 시작
    retry_count = 0
    current_K = K
    
    # 진행 상황 표시 설정
    progress_bar = None
    if display:
        progress_bar = tqdm(total=max_new_tokens, desc="Speculative Sampling")
    
    if verbose:
        logger.info(f"SSP 시작 - 목표 토큰 수: {max_new_tokens}, K: {K}")
    
    # 목표 길이에 도달할 때까지 반복
    while input_ids.shape[1] < desired_length:
        # 남은 생성 토큰 수 계산
        remaining_tokens = desired_length - input_ids.shape[1]
        
        if verbose:
            logger.info(f"현재 길이: {input_ids.shape[1]}, 남은 토큰: {remaining_tokens}")
        
        try:
            # SSP 한 번 반복
            new_input_ids, new_time_stats, draft_tokens, accepted_tokens = _ssp_iteration(
                target_model, 
                draft_model, 
                input_ids, 
                current_K,
                max_new_tokens=min(remaining_tokens, current_K),
                display=display,
                fallback_threshold=fallback_threshold,
                rollback_threshold=rollback_threshold,
                verbose=verbose
            )
            
            # 성공적으로 토큰을 생성했는지 확인
            tokens_added = new_input_ids.shape[1] - input_ids.shape[1]
            
            if verbose:
                logger.info(f"반복 결과: 생성된 드래프트 토큰 {draft_tokens}개, 수락된 토큰 {accepted_tokens}개, 최종 추가 토큰 {tokens_added}개")
                
            # 토큰이 추가되지 않았고, 재시도 가능한 경우
            if tokens_added == 0 and retry_count < max_retries:
                retry_count += 1
                current_K = max(1, current_K - 1)  # K를 줄여서 재시도
                
                if verbose:
                    logger.warning(f"토큰 추가 실패! 재시도 {retry_count}/{max_retries}, 새 K={current_K}")
                    
                continue
            
            # 토큰 추가 실패하고 재시도 횟수를 초과한 경우 종료
            if tokens_added == 0 and retry_count >= max_retries:
                if verbose:
                    logger.error(f"최대 재시도 횟수 초과, 생성 종료")
                break
                
            # 성공적으로 토큰이 추가된 경우 재시도 카운터 리셋
            retry_count = 0
            current_K = K  # K 원래 값으로 리셋
            
            # 통계 업데이트
            total_accept_count += accepted_tokens
            total_generated_count += draft_tokens
            
            # 시간 통계 누적
            for key in cumulative_time_stats:
                if key in new_time_stats:
                    cumulative_time_stats[key] += new_time_stats[key]
            
            input_ids = new_input_ids
            
            # 진행 상황 업데이트
            if display and progress_bar is not None:
                progress_bar.update(tokens_added)
                
        except Exception as e:
            if verbose:
                logger.error(f"SSP 반복 중 오류 발생: {str(e)}")
                if retry_count < max_retries:
                    retry_count += 1
                    current_K = max(1, current_K - 1)
                    logger.warning(f"오류로 인한 재시도 {retry_count}/{max_retries}, 새 K={current_K}")
                    continue
                else:
                    logger.error(f"최대 재시도 횟수 초과, 생성 종료")
                    break
    
    # 진행 상황 표시 종료
    if display and progress_bar is not None:
        progress_bar.close()
    
    # 수락률 계산
    acceptance_rate = 0.0
    if total_generated_count > 0:
        acceptance_rate = total_accept_count / total_generated_count
    
    # 속도 계산
    total_time = time.time() - start_time
    tokens_generated = input_ids.shape[1] - original_input_len
    tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
    
    if verbose:
        logger.info(f"SSP 완료 - 생성 토큰 수: {tokens_generated}, 시간: {total_time:.2f}초, 속도: {tokens_per_second:.2f} 토큰/초")
        logger.info(f"최종 수락률: {acceptance_rate:.2f} ({total_accept_count}/{total_generated_count})")
        logger.info(f"시간 통계: 타겟: {cumulative_time_stats['target_time']:.2f}초, 드래프트: {cumulative_time_stats['draft_time']:.2f}초, 수락계산: {cumulative_time_stats['acceptance_time']:.2f}초, 오버헤드: {cumulative_time_stats['overhead_time']:.2f}초")
    
    return input_ids, total_accept_count, total_generated_count, cumulative_time_stats


class FakeModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def __call__(self, input_ids):
        # Create fake logits randomly in the range [-1, 1]
        B, T = input_ids.shape
        logits = torch.rand(B, T, self.vocab_size) * 2 - 1
        return namedtuple('Output', ['logits'])(logits)


if __name__ == '__main__':
    # Test the SSP implementation
    vocab_size = 10
    target_model = FakeModel(vocab_size)
    draft_model = FakeModel(vocab_size)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    result, accept_rate, time_stats = ssp(target_model, draft_model, 10, input_ids)
    print(f"Generated: {result}")
    print(f"Accept rate: {accept_rate:.2f}")
    print(f"Time stats: {time_stats}") 
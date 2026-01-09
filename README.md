# POGO: Policy Optimization via Gradient flow in Offline RL

POGO는 오프라인 강화학습에서 Gradient flow를 통한 정책 최적화 알고리즘입니다. JKO (Jordan-Kinderlehrer-Otto) chain을 사용하여 여러 actor를 순차적으로 학습하며, Sinkhorn W2 거리를 활용한 정책 간 거리 측정을 구현합니다.

## 주요 특징

- **JKO Chain**: 여러 actor를 순차적으로 학습하는 gradient flow 기반 접근법
- **Transport Map Actor**: `T_s: z ~ N(0,I) → action space` 형태의 정책 표현
- **Sinkhorn W2 Distance**: 정책 간 거리를 Sinkhorn 알고리즘으로 계산
- **재현성 보장**: 모든 랜덤 샘플링에 시드 고정 기능 포함
- **D4RL 지원**: D4RL 벤치마크 데이터셋 지원
- **자동화된 실험**: `config.yaml` 기반 대규모 실험 자동 실행

## 알고리즘 개요

### JKO Chain과 Gradient Flow

POGO는 JKO (Jordan-Kinderlehrer-Otto) chain을 사용하여 gradient flow의 이산 근사를 구현합니다. 여러 actor를 순차적으로 연결하여 연속적인 gradient flow를 근사합니다:

- **Actor 0 (π₀)**: 데이터셋 액션에 대한 L2 loss로 학습
- **Actor i (πᵢ, i ≥ 1)**: 이전 actor (πᵢ₋₁)에 대한 Sinkhorn W2 거리로 학습

각 actor는 gradient flow의 한 단계를 나타내며, 전체 chain은 연속적인 정책 진화를 이산적으로 근사합니다.

**Actor 0가 L2 loss를 사용하는 이유**: 데이터셋의 behavior policy π_β가 delta distribution(점 분포)이면, W2 거리가 L2 거리와 수학적으로 동일합니다. 따라서 Actor 0는 데이터셋 액션에 대해 L2 loss를 사용합니다. 반면 이후 actor들은 이전 actor라는 연속적인 분포를 reference로 하므로, 분포 간 거리를 정확히 측정하는 Sinkhorn W2 거리가 필요합니다.

### 학습 목표

각 actor는 다음 JKO loss를 최소화합니다:

```
L_i = -λ * E[Q(s, πᵢ(s,z))] + w_i * W₂(πᵢ, πᵢ₋₁)
```

여기서:
- `Q(s, a)`: Critic 네트워크의 Q-value
- `W₂(πᵢ, πᵢ₋₁)`: Sinkhorn W2 거리 (i=0일 때는 L2 거리)
- `w_i`: W2 거리의 가중치
- `λ`: Q-value의 정규화 계수

### Transport Map

Actor는 transport map을 neural network로 모델링합니다:

```
π(s, z) = T_s(s, z),  where z ~ N(0, I)
```

여기서 `T_s`는 state와 noise `z`를 입력으로 받아 action을 출력하는 neural network입니다. 이를 통해 같은 state에서도 다른 z 샘플링으로 다양한 액션을 생성할 수 있습니다.

### Critic 학습 (FQL 영향)

Critic은 첫 번째 actor (online actor)를 behavior policy로 사용하여 TD target을 계산합니다. 이는 Flow Q Learning (FQL)에서 영향을 받은 설계입니다:

```
Q_target = r + γ * min(Q₁(s', π₀(s', z')), Q₂(s', π₀(s', z')))
```

Online policy를 사용함으로써 target policy의 지연 업데이트 문제를 완화하고, 더 빠른 학습을 가능하게 합니다.

## 설치

### 요구사항

- Python 3.10+
- PyTorch (CUDA 지원 권장)
- D4RL
- GeomLoss (Sinkhorn 거리 계산용)

### Conda 환경 설정

```bash
conda create -n offrl python=3.10
conda activate offrl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install d4rl geomloss numpy gym pyyaml
```

## 사용 방법

### 단일 실험 실행

```bash
python main.py \
    --env halfcheetah-medium-v2 \
    --seed 0 \
    --max_timesteps 1000000 \
    --eval_freq 10000 \
    --w2_weights 0.2 0.2 0.2 \
    --lr 3e-4 \
    --save_model \
    --checkpoint_dir ./logs/checkpoints
```

### 대규모 실험 자동 실행

`config.yaml` 파일을 수정한 후:

```bash
python run_comparison.py --config config.yaml
```

### Config 파일 구조

`config.yaml`에서 각 환경별로 `w2_weights_base`와 `w2_weights_third`를 설정하면 자동으로 조합이 생성됩니다:

```yaml
environments:
  halfcheetah:
    medium:
      w2_weights_base: [0.2, 0.2]      # Actor 0, 1의 가중치
      w2_weights_third: [0.1, 0.2, 0.3]  # Actor 2의 가중치 (3가지 조합 생성)
      learning_rate: 3e-4
```

이 설정은 다음 3가지 조합을 자동으로 생성합니다:
- `[0.2, 0.2, 0.1]`
- `[0.2, 0.2, 0.2]`
- `[0.2, 0.2, 0.3]`

## 주요 하이퍼파라미터

### Agent 파라미터

- `w2_weights`: 각 actor의 W2 거리 가중치 리스트 (예: `[0.2, 0.2, 0.2]`)
- `lr`: 학습률 (로코모션: `3e-4`, antmaze: `1e-4`)
- `discount`: 할인 계수 (기본값: `0.99`)
- `tau`: Target network soft-update 계수 (기본값: `0.005`)
- `policy_noise`: TD3 스타일 노이즈 (기본값: `0.2`)
- `noise_clip`: 노이즈 클리핑 (기본값: `0.5`)
- `policy_freq`: Policy 업데이트 빈도 (기본값: `1`)

### Sinkhorn 파라미터

- `K`: 각 state당 샘플 수 (기본값: `4`)
- `blur`: Sinkhorn regularization (기본값: `0.05`)
- `backend`: "tensorized", "online", "auto" (기본값: "tensorized")

## 결과 확인

### 로그 파일

학습 로그는 `logs/{env_name}/w2_{weights}/seed_{seed}/training/` 디렉토리에 저장됩니다.

### 평가 결과

각 evaluation step마다 모든 actor의 성능이 출력됩니다:

```
Evaluation over 10 episodes:
  Actor 0 - Deterministic: 60.967, D4RL score: 60.967
  Actor 1 - Deterministic: 58.981, D4RL score: 58.981
  Actor 2 - Deterministic: 65.498, D4RL score: 65.498
```

### 체크포인트

체크포인트는 `logs/{env_name}/w2_{weights}/seed_{seed}/checkpoints/` 디렉토리에 저장되며, 학습 중단 시 재개할 수 있습니다.

## 재현성

모든 랜덤 샘플링에 시드 고정이 적용되어 있습니다:

- **데이터 샘플링**: ReplayBuffer에서 배치 샘플링 시 시드 고정
- **z 샘플링**: Transport map의 z 샘플링 시 시드 고정
- **환경 초기화**: 평가 시 환경 초기화 시드 고정

같은 시드로 실행하면 완전히 동일한 결과를 얻을 수 있습니다.

## 파일 구조

```
POGO_sv/
├── agent.py              # POGO agent 구현 (Actor, Critic, 학습 로직)
├── main.py               # 단일 실험 실행 스크립트
├── run_comparison.py     # 대규모 실험 자동 실행 스크립트
├── utils.py              # ReplayBuffer 및 유틸리티 함수
├── config.yaml           # 실험 설정 파일
├── logs/                 # 학습 로그 및 체크포인트
└── results/              # 평가 결과 저장
```

## 지원 환경

### 로코모션 환경
- `halfcheetah-medium-v2`, `halfcheetah-medium-replay-v2`, `halfcheetah-medium-expert-v2`
- `hopper-medium-v2`, `hopper-medium-replay-v2`, `hopper-medium-expert-v2`
- `walker2d-medium-v2`, `walker2d-medium-replay-v2`, `walker2d-medium-expert-v2`

### Antmaze 환경
- `antmaze-umaze-v2`, `antmaze-umaze-diverse-v2`
- `antmaze-medium-play-v2`, `antmaze-medium-diverse-v2`
- `antmaze-large-play-v2`, `antmaze-large-diverse-v2`

## 참고 문헌

자세한 알고리즘 설명은 `POGO.pdf`와 `POGO_supplement.pdf`를 참고하세요.

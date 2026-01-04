# File: pogogo/lagged/main.py

#!/usr/bin/env python3
"""
POGO: Single-step / Two-step 실행 메인 (단순화)
- 사용법은 질문의 1~4번 시나리오와 완전히 호환됩니다.
"""

import os
import json
import argparse
from datetime import datetime
import numpy as np
import torch
import gym
import d4rl

try:
    import wandb
except ImportError:
    wandb = None

import utils
from agent import POGO


# ---------------------------
# 유틸
# ---------------------------
def set_global_seed(env, seed: int):
    """Gym 버전 호환 시드 설정."""
    try:
        env.reset(seed=seed)
    except TypeError:
        # old gym
        env.seed(seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    set_global_seed(env, seed)
    return env


def normalize(replay_buffer, enable: bool):
    if enable:
        mean, std = replay_buffer.normalize_states()
    else:
        sdim = replay_buffer.state.shape[1]
        mean = np.zeros((1, sdim), dtype=np.float32)
        std  = np.ones((1, sdim), dtype=np.float32)
    return mean, std


# ---------------------------
# 평가 루틴 (환경 1개 재사용)
# ---------------------------
@torch.no_grad()
def eval_policy(policy, eval_env, mean, std, base_seed, eval_episodes=10, deterministic=True, actor_idx=None):
    """정책 평가: deterministic과 stochastic 모두 평가하고 결과 반환
    
    Args:
        actor_idx: 평가할 actor 인덱스 (None이면 마지막 actor)
    
    Returns:
        tuple: (det_avg, det_score, stoch_avg, stoch_score)
            - det_avg: deterministic 평균 리워드
            - det_score: deterministic D4RL 정규화 점수
            - stoch_avg: stochastic 평균 리워드
            - stoch_score: stochastic D4RL 정규화 점수
    """
    
    # Deterministic 평가
    det_total = 0.0
    for ep in range(eval_episodes):
        try:
            reset_result = eval_env.reset(seed=base_seed + ep)
        except TypeError:
            reset_result = eval_env.reset()

        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        done = False
        ep_ret = 0.0
        while not done:
            nstate = (np.asarray(state).reshape(1, -1) - mean) / std
            action = policy.select_action(nstate, deterministic=True, actor_idx=actor_idx)
            step_out = eval_env.step(action)
            if len(step_out) == 5:
                state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                state, reward, done, _ = step_out
            ep_ret += reward
        det_total += ep_ret
    
    # Stochastic 평가
    stoch_total = 0.0
    for ep in range(eval_episodes):
        try:
            reset_result = eval_env.reset(seed=base_seed + ep)
        except TypeError:
            reset_result = eval_env.reset()

        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        done = False
        ep_ret = 0.0
        while not done:
            nstate = (np.asarray(state).reshape(1, -1) - mean) / std
            action = policy.select_action(nstate, deterministic=False, actor_idx=actor_idx)
            step_out = eval_env.step(action)
            if len(step_out) == 5:
                state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                state, reward, done, _ = step_out
            ep_ret += reward
        stoch_total += ep_ret
    
    # 결과 계산
    det_avg = det_total / eval_episodes
    stoch_avg = stoch_total / eval_episodes
    det_score = eval_env.get_normalized_score(det_avg) * 100.0
    stoch_score = eval_env.get_normalized_score(stoch_avg) * 100.0
    
    # 결과 반환 (det_avg, det_score, stoch_avg, stoch_score)
    return det_avg, det_score, stoch_avg, stoch_score


def final_evaluation(policy, env_name, seed, mean, std, runs=5, episodes=10, actor_idx=None, use_wandb=True):
    eval_env = make_env(env_name, seed + 10_000)  # 훈련 시드와 멀리 떨어뜨림

    det_scores, stoch_scores = [], []
    for r in range(runs):
        _, det_score, _, _ = eval_policy(
            policy, eval_env, mean, std,
            base_seed=1000 + 100 * r, eval_episodes=episodes, deterministic=True,
            actor_idx=actor_idx
        )
        _, _, _, stoch_score = eval_policy(
            policy, eval_env, mean, std,
            base_seed=2000 + 100 * r, eval_episodes=episodes, deterministic=False,
            actor_idx=actor_idx
        )
        det_scores.append(det_score)
        stoch_scores.append(stoch_score)

    det_scores = np.array(det_scores, dtype=np.float32)
    stoch_scores = np.array(stoch_scores, dtype=np.float32)

    print("======== Final Evaluation (trained weights) ========")
    print(f"[FINAL] Deterministic: mean={det_scores.mean():.3f}, std={det_scores.std():.3f} over {runs}x{episodes}")
    print(f"[FINAL] Stochastic:   mean={stoch_scores.mean():.3f}, std={stoch_scores.std():.3f} over {runs}x{episodes}")
    
    # Log final evaluation to wandb
    if use_wandb and wandb is not None:
        actor_suffix = f"_actor_{actor_idx}" if actor_idx is not None else ""
        wandb.log({
            f"final{actor_suffix}/det_mean": float(det_scores.mean()),
            f"final{actor_suffix}/det_std": float(det_scores.std()),
            f"final{actor_suffix}/stoch_mean": float(stoch_scores.mean()),
            f"final{actor_suffix}/stoch_std": float(stoch_scores.std()),
            f"final{actor_suffix}/runs": runs,
            f"final{actor_suffix}/episodes": episodes,
        })
    
    return det_scores, stoch_scores


# ---------------------------
# 체크포인트 유틸
# ---------------------------
def save_checkpoint(agent, ckpt_dir: str, prefix: str, step: int, phase: str, extra_meta=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, prefix)
    agent.save(path)
    meta = {
        "step": int(step),
        "phase": phase,
        "total_it": int(getattr(agent, "total_it", step)),
        "checkpoint_name": prefix,
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(path + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[CKPT] Saved: {path}_* (step={step}, phase={phase})")


def load_checkpoint_into_AgentA(agentA, load_prefix: str):
    print(f"[LOAD] Loading checkpoint from: {load_prefix}")
    agentA.load(load_prefix)
    meta_path = load_prefix + "_meta.json"
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        total_it = metadata.get("total_it")
        if total_it is None:
            total_it = metadata.get("step", 0)
        if total_it is not None:
            agentA.total_it = int(total_it)
        print(f"[LOAD] Done. (step={metadata.get('step', 'unknown')}, phase={metadata.get('phase', 'unknown')})")
    else:
        print("[LOAD] Metadata not found alongside checkpoint. Assuming fresh start from weights only.")
    return metadata


# ---------------------------
# 통합 학습 (POGO: One-step + Two-step 동시 진행)
# ---------------------------
def train_unified(agent, env_name, seed, replay_buffer, mean, std,
                  max_steps, eval_freq, save_model, file_name, ckpt_dir,
                  midpoint_step, start_step=0, use_wandb=True):
    """
    POGO 통합 학습: actor_one과 actor_two를 동시에 학습
    두 actor 모두 평가하고 각각의 로그를 저장
    """
    eval_env = make_env(env_name, seed + 1234)
    eval_file_one = f"./results/{file_name}_actor_one.npy"
    eval_file_two = f"./results/{file_name}_actor_two.npy"
    
    if start_step > 0 and os.path.exists(eval_file_one):
        evaluations_one = list(np.load(eval_file_one))
    else:
        evaluations_one = []
    
    if start_step > 0 and os.path.exists(eval_file_two):
        evaluations_two = list(np.load(eval_file_two))
    else:
        evaluations_two = []

    midpoint_target = min(max_steps, int(midpoint_step)) if midpoint_step is not None else None
    midpoint_saved = midpoint_target is None or start_step >= midpoint_target

    num_actors = getattr(agent, "num_actors", 1)
    w2_weights = getattr(agent, "w2_weights", [1.0] * num_actors)
    w2_str = ", ".join([f"{w:.3f}" for w in w2_weights])
    print(f"🚀 통합 학습 시작: {start_step} ~ {max_steps-1} steps (POGO, {num_actors}개 actor)")
    print(f"   Actor weights: [{w2_str}]")
    
    for global_step in range(start_step, max_steps):
        metrics = agent.train(replay_buffer, batch_size=256)
        
        # Log training metrics to wandb
        if use_wandb and wandb is not None:
            log_dict = {
                "train/timestep": global_step + 1,
            }
            # Only log metrics that exist (don't log 0.0 for missing metrics)
            # Critic is trained every step (unless freeze_critic)
            if "critic_loss" in metrics:
                log_dict["train/critic_loss"] = metrics["critic_loss"]
            
            # Actor is updated every policy_freq steps, so these metrics are sparse
            # Only log when actor was actually updated
            num_actors = getattr(agent, "num_actors", 1)
            for i in range(num_actors):
                if f"actor_{i}_loss" in metrics:
                    log_dict.update({
                        f"train/actor_{i}_loss": metrics[f"actor_{i}_loss"],
                        f"train/Q_{i}_mean": metrics[f"Q_{i}_mean"],
                        f"train/w2_{i}_distance": metrics[f"w2_{i}_distance"],
                    })
            wandb.log(log_dict, step=global_step + 1)

        if (global_step + 1) % eval_freq == 0:
            print(f"[Training] Time steps: {global_step + 1}")
            # 모든 actor 평가
            num_actors = agent.num_actors
            actor_results = []
            
            for i in range(num_actors):
                det_avg, det_score, stoch_avg, stoch_score = eval_policy(
                    agent, eval_env, mean, std, 
                    base_seed=100 + i * 100, eval_episodes=10, 
                    deterministic=True,
                    actor_idx=i
                )
                actor_results.append({
                    'det_avg': det_avg,
                    'det_score': det_score,
                    'stoch_avg': stoch_avg,
                    'stoch_score': stoch_score
                })
                
                # 첫 번째와 두 번째 actor는 기존 파일에 저장 (하위 호환성)
                if i == 0:
                    evaluations_one.append(det_score)
                    np.save(eval_file_one, evaluations_one)
                elif i == 1:
                    evaluations_two.append(det_score)
                    np.save(eval_file_two, evaluations_two)
            
            # 모든 actor의 결과 출력
            print("---------------------------------------")
            print("Evaluation over 10 episodes:")
            for i in range(num_actors):
                r = actor_results[i]
                print(f"  Actor {i} - Deterministic: {r['det_avg']:.3f}, D4RL score: {r['det_score']:.3f}")
                print(f"  Actor {i} - Stochastic: {r['stoch_avg']:.3f}, D4RL score: {r['stoch_score']:.3f}")
            print("---------------------------------------")
            
            # Log eval results to wandb
            if use_wandb and wandb is not None:
                eval_log_dict = {"eval/timestep": global_step + 1}
                for i in range(num_actors):
                    r = actor_results[i]
                    eval_log_dict.update({
                        f"eval/actor_{i}_det_return": r['det_avg'],
                        f"eval/actor_{i}_det_score": r['det_score'],
                        f"eval/actor_{i}_stoch_return": r['stoch_avg'],
                        f"eval/actor_{i}_stoch_score": r['stoch_score'],
                    })
                wandb.log(eval_log_dict, step=global_step + 1)

        if not midpoint_saved and midpoint_target is not None and (global_step + 1) >= midpoint_target:
            checkpoint_step = midpoint_target
            mid_name = f"{file_name}_mid_{checkpoint_step}"
            save_checkpoint(
                agent,
                ckpt_dir,
                mid_name,
                step=checkpoint_step,
                phase="unified",
                extra_meta={
                    "file_name": file_name,
                    "max_timesteps": max_steps,
                    "split_ratio": midpoint_target / max_steps if max_steps else 0.0,
                    "env": env_name,
                    "seed": seed,
                },
            )
            midpoint_saved = True

    return agent


# ---------------------------
# main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # Experiment
    p.add_argument("--env", default="hopper-medium-v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_freq", type=int, default=5000)
    p.add_argument("--max_timesteps", type=int, default=1_000_000)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--normalize", default=True, type=bool)

    # Shared/TD3
    p.add_argument("--batch_size", type=int, default=256)  # agent 내부에서 사용
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--policy_noise", type=float, default=0.2)
    p.add_argument("--noise_clip", type=float, default=0.5)
    p.add_argument("--policy_freq", type=int, default=2)

    # POGO
    p.add_argument("--w2_weights", type=float, nargs="+", default=[0.5, 0.5], 
                   help="W2 weights for each actor (예: --w2_weights 0.2 0.2)")
    p.add_argument("--lr", type=float, default=3e-4)

    # Final eval
    p.add_argument("--final_eval_runs", type=int, default=5)
    p.add_argument("--final_eval_episodes", type=int, default=10)

    # Unified training control
    p.add_argument("--checkpoint_dir", type=str, default="./logs/checkpoints")

    p.add_argument("--start_mode", choices=["scratch", "load"], default="scratch",
                   help="scratch: 0→max 전부 진행 / load: 체크포인트에서 이어서 계속")
    p.add_argument("--load_prefix", type=str, default="",
                   help="체크포인트 prefix (확장자 없이). 예: ./logs/checkpoints/POGO_hopper-medium-v2_0_mid_500000")

    # Wandb
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb_project", type=str, default="POGOGO", help="Wandb project name")
    p.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity name")
    p.add_argument("--wandb_name", type=str, default=None, help="Wandb run name (default: env_seed)")

    return p.parse_args()


def main():
    args = parse_args()

    # Wandb 초기화
    if args.wandb and wandb is not None:
        # 날짜+시간 형식: YYYYMMDD_HHMMSS
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_name = args.wandb_name or f"{args.env}_seed{args.seed}_{datetime_str}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_name,
            config={
                "env": args.env,
                "seed": args.seed,
                "max_timesteps": args.max_timesteps,
                "eval_freq": args.eval_freq,
                "batch_size": args.batch_size,
                "discount": args.discount,
                "tau": args.tau,
                "policy_noise": args.policy_noise,
                "noise_clip": args.noise_clip,
                "policy_freq": args.policy_freq,
                "w2_weights": args.w2_weights,
                "lr": args.lr,
                "normalize": args.normalize,
            },
        )
        use_wandb = True
    else:
        use_wandb = False
        if args.wandb and wandb is None:
            print("Warning: wandb requested but not installed. Continuing without wandb logging.")

    # 파일 이름 기본값 (resume 시 덮어쓰기)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_file_name = f"POGO_{args.env}_{args.seed}_{timestamp}"
    file_name = default_file_name
    os.makedirs("./results", exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 환경 및 데이터셋
    env = make_env(args.env, args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 데이터 적재
    rb = utils.ReplayBuffer(state_dim, action_dim)
    rb.convert_D4RL(d4rl.qlearning_dataset(env))
    mean, std = normalize(rb, args.normalize)

    # POGO 통합 Agent 생성
    agent = POGO(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise * max_action,
        noise_clip=args.noise_clip * max_action,
        policy_freq=args.policy_freq,
        w2_weights=args.w2_weights,
        lr=args.lr,
    )

    # 로드 옵션
    resume_metadata = {}
    if args.start_mode == "load" and args.load_prefix:
        resume_metadata = load_checkpoint_into_AgentA(agent, args.load_prefix)
        if resume_metadata:
            loaded_name = resume_metadata.get("file_name")
            if loaded_name:
                file_name = loaded_name

    resume_step = int(resume_metadata.get("step", 0))

    # 스케줄 (midpoint는 체크포인트 저장용)
    midpoint_step = min(args.max_timesteps, int(round(args.max_timesteps * 0.5)))

    # 통합 학습
    if args.start_mode == "load" and resume_step > 0:
        print(f"🔁 학습 재개: step {resume_step} → {args.max_timesteps}")
        agent = train_unified(
            agent, args.env, args.seed, rb, mean, std,
            max_steps=args.max_timesteps, eval_freq=args.eval_freq,
            save_model=args.save_model, file_name=file_name,
            ckpt_dir=args.checkpoint_dir, midpoint_step=midpoint_step,
            start_step=resume_step, use_wandb=use_wandb
        )
    else:
        # scratch 모드: 처음부터 전체 학습
        agent = train_unified(
            agent, args.env, args.seed, rb, mean, std,
            max_steps=args.max_timesteps, eval_freq=args.eval_freq,
            save_model=args.save_model, file_name=file_name,
            ckpt_dir=args.checkpoint_dir, midpoint_step=midpoint_step,
            start_step=0, use_wandb=use_wandb
        )

    # -------- Final evaluation for ALL actors --------
    num_actors = getattr(agent, "num_actors", 1)
    print("\n======== Final Evaluation (all actors) ========")
    for i in range(num_actors):
        print(f"\n======== Final Evaluation: Actor {i} ========")
        final_evaluation(
            agent,
            args.env,
            args.seed,
            mean,
            std,
            runs=args.final_eval_runs,
            episodes=args.final_eval_episodes,
            actor_idx=i,
            use_wandb=use_wandb,
        )
    
    # Finish wandb run
    if use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

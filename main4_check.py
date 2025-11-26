# File: main4.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import gym
import argparse
import d4rl

import utils

try:
    import wandb
except ImportError:
    wandb = None

from agent_gflow import GFlow_W2_Refine, device


def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            nstate = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(nstate)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100.0
    print("---------------------------------------")
    print(f"[EVAL] {eval_episodes} episodes: {avg_reward:.3f}, D4RL: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument("--env", default="halfcheetah-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=5_000)
    parser.add_argument("--max_timesteps", type=int, default=500_000)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--normalize", default=True, type=bool)

    # TD3-style shared params
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau_soft", type=float, default=0.005)   # target soft-update
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    # GFlow hyperparams
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--w2_weight", type=float, default=0.5)

    # critic checkpoint (1M one-step run)
    parser.add_argument("--critic_path", default=None)

    parser.add_argument("--final_eval_runs", type=int, default=5)
    parser.add_argument("--final_eval_episodes", type=int, default=10)

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="aistats_tau")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default="")

    args = parser.parse_args()

    num_steps = 4  
    file_name = f"POGO_{args.env}_seed{args.seed}_multi-step{num_steps}"

    print("======================================")
    print(f"[SETUP] 4-step refine, env={args.env}, seed={args.seed}")
    print(f"        base w2={args.w2_weight}, per-step w2={args.w2_weight*num_steps}")
    print("======================================")

    os.makedirs("./POGO-W/results", exist_ok=True)
    os.makedirs("./checkpoint", exist_ok=True)

    # W&B init
    run = None
    if args.wandb:
        run_name = args.run_name or file_name
        init_kwargs = dict(
            project=args.wandb_project,
            name=run_name,
            group=args.env,
            tags=["POGO_multi-step4"],
            config=vars(args),
        )
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        run = wandb.init(**init_kwargs)
        wandb.log({"_boot": 1, "timesteps": 0}, step=0)

    if args.wandb and wandb.run is not None:
        for k, v in dict(wandb.config).items():
            if hasattr(args, k):
                setattr(args, k, v)

    # Env & dataset
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean = np.zeros((1, state_dim), dtype=np.float32)
        std = np.ones((1, state_dim), dtype=np.float32)

    # Agent
    agent = GFlow_W2_Refine(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau_soft,
        policy_noise=args.policy_noise * max_action,
        noise_clip=args.noise_clip * max_action,
        policy_freq=args.policy_freq,
        alpha=args.alpha,
        w2_weight=args.w2_weight * num_steps,
        entropy_weight=args.entropy_weight,
        learning_rate=args.learning_rate,
    )

    actor_path = f"critics/POGO-W_One-Step_{args.env}_{args.seed}_actor"
    print(f"[LOAD] actor from {actor_path}")
    actor_state = torch.load(actor_path, map_location=device)
    agent.actor.load_state_dict(actor_state)
    agent.actor_target.load_state_dict(actor_state)
    agent.behavior_policy.load_state_dict(actor_state)

    critic_path = f"critics/POGO-W_One-Step_{args.env}_{args.seed}_critic"
    print(f"[LOAD] critic from {critic_path}")
    critic_state = torch.load(critic_path, map_location=device)
    agent.critic.load_state_dict(critic_state)
    agent.critic_target.load_state_dict(critic_state)
    print("[LOAD] critic loaded. Freezing critic / critic_target params.")
    for p in agent.critic.parameters():
        p.requires_grad = False
    for p in agent.critic_target.parameters():
        p.requires_grad = False

    per_step_w2 = args.w2_weight * num_steps
    evaluations = []
    best_eval = -np.inf
    global_step = 0

    print("==== 4-step schedule ====")
    print(f"  per-step w2 = {per_step_w2}")
    print(f"  steps per stage = {args.max_timesteps}")
    print("=========================")

    for stage in range(num_steps):
        if stage != 0:
            agent.actor.load_state_dict(actor_state)
            agent.actor_target.load_state_dict(actor_state)
            agent.behavior_policy.load_state_dict(actor_state)
            agent.critic.load_state_dict(critic_state)
            agent.critic_target.load_state_dict(critic_state)

        print(f"\n========== Stage {stage+1}/{num_steps} ==========")
        print(f"  using w2_weight = {per_step_w2:.6f}")
        agent.w2_weight = per_step_w2

        for s in range(args.max_timesteps):
            metrics = agent.train(replay_buffer, args.batch_size)
            global_step += 1

            if args.wandb:
                log_data = {
                    "timesteps": global_step,
                    "stage": stage,
                    "w2_weight": per_step_w2,
                }
                for k in ["critic_loss", "actor_loss", "behavior_loss"]:
                    if k in metrics:
                        name = "train/behavior_nll" if k == "behavior_loss" else f"train/{k}"
                        log_data[name] = metrics[k]
                for k in ["lambda", "Q_mean", "w2_distance", "entropy",
                          "actor_std", "behavior_std", "actor_mean", "behavior_mean"]:
                    if k in metrics:
                        log_data[f"train/{k}"] = metrics[k]
                wandb.log(log_data, step=global_step)

            # if global_step % args.eval_freq == 0:
            #     print(f"[Stage {stage}] timesteps={global_step}")
            #     d4rl_score = eval_policy(agent, args.env, args.seed, mean, std)
            #     evaluations.append(d4rl_score)
            #     np.save(f"./POGO-W/results/{file_name}", evaluations)
            #     if args.wandb:
            #         wandb.log(
            #             {"eval/d4rl": d4rl_score,
            #              "timesteps": global_step,
            #              "stage": stage},
            #             step=global_step,
            #         )
            #     if args.save_model:
            #         agent.save(f"./POGO-W/models/{file_name}")
            #         if d4rl_score > best_eval:
            #             best_eval = d4rl_score
            #             agent.save(f"./POGO-W/models/{file_name}_best")

        actor_state = agent.actor.state_dict()
        critic_state = agent.critic.state_dict()

        print("======== Multi-step Evaluation (4-step refine) ========")
        final_scores = []
        for r in range(args.final_eval_runs):
            score = eval_policy(
                agent, args.env, args.seed, mean, std,
                seed_offset=1000 + 100 * r,
                eval_episodes=args.final_eval_episodes
            )
            final_scores.append(score)
        final_scores = np.array(final_scores, dtype=np.float32)
        final_mean, final_std = float(final_scores.mean()), float(final_scores.std())
        print(f"[FINAL] mean={final_mean:.3f}, std={final_std:.3f}")
        if wandb and args.wandb:
            wandb.log({
                "multi/eval_mean": final_mean,
                "multi/eval_std": final_std},
                step = stage)

        agent.save(f"./checkpoint/{file_name}_stage{stage}")

    # Final evaluation
    print("======== Final Evaluation (4-step refine) ========")
    final_scores = []
    for r in range(args.final_eval_runs):
        score = eval_policy(
            agent, args.env, args.seed, mean, std,
            seed_offset=1000 + 100 * r,
            eval_episodes=args.final_eval_episodes
        )
        final_scores.append(score)
    final_scores = np.array(final_scores, dtype=np.float32)
    final_mean, final_std = float(final_scores.mean()), float(final_scores.std())
    print(f"[FINAL] mean={final_mean:.3f}, std={final_std:.3f}")
    if wandb and args.wandb:
        wandb.log({
            "final/eval_mean": final_mean,
            "final/eval_std": final_std,
        })
        wandb.finish()

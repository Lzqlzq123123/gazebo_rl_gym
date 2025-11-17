#!/usr/bin/env python3
"""
Minimal evaluation script for a single-run Gazebo policy saved by OnPolicyRunner.
Usage:
    python eval_rsl_ppo_single.py \
        --train-cfg src/gazebo_rl_gym/config/train/warehouse_ppo_turtlebot3.yaml \
        --model logs/warehouse_ppo_turtlebot3/model_100.pt \
        --num-episodes 5
"""

import argparse
import os
from typing import Any, Dict

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.runners import OnPolicyRunner
from gazebo_rl_gym.envs.gazebo_vec_env import GazeboSingleVecEnv
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cfg", default="src/gazebo_rl_gym/config/train/warehouse_ppo_turtlebot3.yaml")
    parser.add_argument("--model", required=True, help="Path to saved model (e.g. logs/.../model_XXX.pt)")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", default="logs/warehouse_ppo_turtlebot3/eval_logs", help="Optional logdir for evaluation metrics (tensorboard)")
    args = parser.parse_args()

    train_cfg = load_yaml(args.train_cfg)
    env_cfg_path = os.path.abspath(train_cfg["env_cfg"])

    device = args.device
    env = GazeboSingleVecEnv(env_cfg_path=env_cfg_path, device=device)

    # Build minimal runner config re-using training config so model structure matches
    runner_cfg: Dict[str, Any] = {
        "num_steps_per_env": train_cfg["num_steps_per_env"],
        "save_interval": train_cfg["save_interval"],
        "logger": train_cfg.get("logger", "tensorboard"),
        "algorithm": train_cfg["algorithm"],
        "policy": train_cfg["policy"],
        "obs_groups": train_cfg["obs_groups"],
    }

    runner = OnPolicyRunner(env=env, train_cfg=runner_cfg, log_dir=None, device=device)

    # Load model; map location to CPU/GPU depending on args.device
    print(f"Loading model from {args.model} ...")
    runner.load(args.model, load_optimizer=False, map_location=args.device)

    # Get inference callable
    infer_fn = runner.get_inference_policy(device=args.device)

    # Optional: create tensorboard writer to log evaluation metrics
    writer = None
    if args.logdir is not None:
        writer = SummaryWriter(log_dir=args.logdir)

    # Run num-episodes and collect stats
    rew_hist = []
    len_hist = []
    for ep in range(args.num_episodes):
        obs = env.reset()  # TensorDict
        done = False
        episode_reward = 0.0
        episode_len = 0
        while not done:
            with torch.no_grad():
                action = infer_fn(obs)  # returns shape (1, action_dim) torch tensor
            obs, reward, done, extras = env.step(action.to(env.device))
            episode_reward += float(reward.item())
            episode_len += 1
        rew_hist.append(episode_reward)
        len_hist.append(episode_len)
        print(f"Episode {ep}: reward={episode_reward:.2f}, length={episode_len}")

        # Optionally write to tensorboard
        if writer is not None:
            writer.add_scalar("Eval/episode_reward", episode_reward, ep)
            writer.add_scalar("Eval/episode_length", episode_len, ep)

    # Summary
    import statistics
    print("Evaluation summary:")
    print(f"Mean reward: {statistics.mean(rew_hist):.2f} ± {statistics.pstdev(rew_hist):.2f}")
    print(f"Mean length: {statistics.mean(len_hist):.2f} ± {statistics.pstdev(len_hist):.2f}")

    if writer is not None:
        writer.close()
    env.close()


if __name__ == "__main__":
    main()
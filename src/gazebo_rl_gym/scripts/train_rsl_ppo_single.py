from __future__ import annotations

"""Entry-point script for training a single-robot Gazebo env with rsl_rl PPO.

Usage (from the gazebo_rl_gym_ws root, inside your rl conda env):

    python src/gazebo_rl_gym/scripts/train_rsl_ppo_single.py \
        --train-cfg src/gazebo_rl_gym/config/train/warehouse_ppo_turtlebot3.yaml

This script:
- Loads a training YAML (algorithm + policy + runner settings + env_cfg path).
- Constructs a GazeboSingleVecEnv from env_cfg.
- Runs rsl_rl.runners.OnPolicyRunner with PPO and logs to Tensorboard.
"""

import argparse
import os
from typing import Any, Dict

import torch
import yaml

from rsl_rl.runners import OnPolicyRunner
from gazebo_rl_gym.envs.gazebo_vec_env import GazeboSingleVecEnv


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-cfg",
        type=str,
        default="src/gazebo_rl_gym/config/train/warehouse_ppo_turtlebot3.yaml",
        help="Path to the training configuration YAML.",
    )
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--model", type=str, default=None, help="Optional path to a saved model to load before training")
    parser.add_argument(
        "--load-optimizer",
        action="store_true",
        help="Whether to load the optimizer state from the saved model (useful when resuming training).",
    )
    parser.add_argument(
        "--model-map-location",
        type=str,
        default=None,
        help="Map location for loading the model (e.g., 'cpu' or 'cuda:0'). If unspecified, auto-detect.",
    )

    args = parser.parse_args()

    train_cfg_path = os.path.abspath(args.train_cfg)
    cfg = load_yaml(train_cfg_path)

    env_cfg_path = os.path.abspath(cfg["env_cfg"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Create VecEnv
    env = GazeboSingleVecEnv(env_cfg_path=env_cfg_path, device=device)

    # 2) Build runner config for OnPolicyRunner
    runner_cfg: Dict[str, Any] = {
        "num_steps_per_env": cfg["num_steps_per_env"],
        "save_interval": cfg["save_interval"],
        "logger": cfg.get("logger", "tensorboard"),
        "algorithm": cfg["algorithm"],
        "policy": cfg["policy"],
        "obs_groups": cfg["obs_groups"],
    }

    # 3) Log directory
    log_root = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, cfg.get("name", "gazebo_single"))
    os.makedirs(log_dir, exist_ok=True)

    # 4) Create runner and start learning
    runner = OnPolicyRunner(env=env, train_cfg=runner_cfg, log_dir=log_dir, device=device)

    # Optional: load a model before training to resume (always treat cfg['num_iterations'] as total)
    cfg_num_iterations = int(cfg.get("num_iterations", 1000))
    if args.model is not None:
        map_location = args.model_map_location or device
        print(f"Loading model from {args.model} (map_location={map_location}) ...")
        runner.load(args.model, load_optimizer=args.load_optimizer, map_location=map_location)
        # Compute how many iterations remain to reach the target total
        remaining = cfg_num_iterations - int(runner.current_learning_iteration)
        if remaining <= 0:
            print(
                f"Config 'num_iterations'={cfg_num_iterations} <= current iteration {runner.current_learning_iteration}. Nothing to run."
            )
            return
        num_learning_iterations = remaining
    else:
        num_learning_iterations = cfg_num_iterations

    runner.learn(num_learning_iterations=num_learning_iterations)


if __name__ == "__main__":
    main()

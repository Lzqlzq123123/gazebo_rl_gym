"""Helpers for loading and validating multi-robot Gazebo configuration files."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from gazebo_rl_gym.envs.registry import robot_registry
from gazebo_rl_gym.utils.path_utils import resolve_relative


@dataclass
class RobotScenario:
    name: str
    preset: str
    pose: Dict[str, float]
    spec: Any


@dataclass
class EnvironmentConfig:
    world_file: str
    robots: List[RobotScenario]
    name: Optional[str] = None
    reward: Dict[str, Any] = field(default_factory=dict)
    termination: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reward_debug: bool = False


def _resolve_world_path(world_entry: Optional[str], *, config_dir: Optional[str] = None) -> str:
    default_world = resolve_relative("worlds", "empty.world")
    if not world_entry:
        return default_world

    if os.path.isabs(world_entry):
        return world_entry

    candidates = []
    if config_dir:
        candidates.append(os.path.join(config_dir, world_entry))
    candidates.append(resolve_relative("worlds", world_entry))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return world_entry


def _build_robot_overrides(entry: Dict[str, Any], global_reward: Dict[str, Any] = None, global_termination: Dict[str, Any] = None, global_observation: Dict[str, Any] = None, global_reward_debug: bool = False) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    for key in ("model", "controller_type", "base_frame", "map_frame"):
        if key in entry:
            overrides[key] = entry[key]

    # 合并全局和机器人特定的 observation 配置
    # 机器人特定的配置优先级更高
    if global_observation or "observation" in entry:
        merged_observation = {}
        if global_observation:
            merged_observation.update(global_observation)
        if "observation" in entry:
            merged_observation.update(entry["observation"])
        if merged_observation:
            overrides["observation"] = merged_observation
    
    # 合并全局和机器人特定的 reward/termination 配置
    # 机器人特定的配置优先级更高
    if global_reward or "reward" in entry:
        merged_reward = {}
        if global_reward:
            merged_reward.update(global_reward)
        if "reward" in entry:
            merged_reward.update(entry["reward"])
        if merged_reward:
            overrides["reward"] = merged_reward
    
    if global_termination or "termination" in entry:
        merged_termination = {}
        if global_termination:
            merged_termination.update(global_termination)
        if "termination" in entry:
            merged_termination.update(entry["termination"])
        if merged_termination:
            overrides["termination"] = merged_termination

    topics = {}
    if "cmd_topic" in entry:
        topics["cmd"] = entry["cmd_topic"]
    if "pose_topic" in entry:
        topics["pose"] = entry["pose_topic"]
    if "scan_topic" in entry:
        topics["scan"] = entry["scan_topic"]
    if topics:
        overrides["topics"] = topics

    if "task" in entry:
        overrides["task"] = entry["task"]
        print(f"Debug - Adding task from entry: {entry['task']}")
        
    # Add global reward_debug
    overrides["reward_debug"] = global_reward_debug

    return overrides


def load_environment_config(path: str) -> EnvironmentConfig:
    # Load environment YAML
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    config_dir = os.path.dirname(os.path.abspath(path))

    world_entry = raw.get("world") or raw.get("world_file") or raw.get("world_name")
    world_file = _resolve_world_path(world_entry, config_dir=config_dir)

    # 提取全局的 observation, reward 和 termination 配置
    global_observation = raw.get("observation", {}) or {}
    global_reward = raw.get("reward", {}) or {}
    global_termination = raw.get("termination", {}) or {}
    global_reward_debug = raw.get("reward_debug", False)

    robots: List[RobotScenario] = []
    for entry in raw.get("robots", []):
        if "preset" not in entry:
            raise ValueError("Robot entry must define a 'preset'")
        if "name" not in entry:
            raise ValueError("Robot entry must define a 'name'")

        print(f"Debug - Processing robot entry: {entry}")
        overrides = _build_robot_overrides(entry, global_reward, global_termination, global_observation, global_reward_debug)
        print(f"Debug - Robot overrides: {overrides}")
        # Create robot specification from registry with RobotCfg
        spec = robot_registry.create(entry["preset"], entry["name"], overrides=overrides)
        robots.append(
            RobotScenario(
                name=entry["name"],
                preset=entry["preset"],
                pose=entry.get("pose", {}),
                spec=spec,
            )
        )

    return EnvironmentConfig(
        world_file=world_file,
        robots=robots,
        name=raw.get("name"),
        reward=global_reward,
        termination=global_termination,
        metadata=raw.get("metadata", {}) or {},
        reward_debug=global_reward_debug,
    )

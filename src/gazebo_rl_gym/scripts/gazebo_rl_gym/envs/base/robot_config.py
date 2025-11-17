from typing import List

from .base_config import BaseConfig


class RobotCfg(BaseConfig):
    """机器人配置类，定义机器人的基本参数、话题、观测、动作空间、奖励和终止条件。"""

    # 机器人模型名称
    model: str = ""
    # 控制器类型
    controller_type: str = ""
    # 机器人基座坐标系模板，使用 {name} 占位符
    base_frame: str = "{name}/base_link"
    # 地图坐标系
    map_frame: str = "map"

    class topics:
        """定义机器人相关的话题名称。"""
        # 速度命令话题
        cmd = "/{name}/cmd_vel"
        # 姿态话题
        pose = "/{name}/ground_truth/state"
        # 激光扫描话题
        scan = "/{name}/scan"
        # 接触状态话题
        contact = "/{name}/contact_state"

    class action_space:
        """动作空间配置。"""
        # 动作下限 [线速度, 角速度]
        low: List[float] = [-0.26, -1.82]
        # 动作上限 [线速度, 角速度]
        high: List[float] = [0.26, 1.82]

    # 注：observation, reward 和 termination 配置应在 YAML 中定义


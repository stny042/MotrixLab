# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
ANYmal C 四足机器人导航环境配置文件

本文件定义了环境的所有可配置参数，包括：
- 传感器噪声配置
- 控制参数配置
- 初始状态配置
- 导航命令范围
- 观测值归一化系数
- 机器人资产信息
- 奖励函数权重
"""

import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

# MuJoCo 场景文件路径（包含机器人模型和环境）
model_file = os.path.dirname(__file__) + "/xmls/scene.xml"


@dataclass
class NoiseConfig:
    """
    传感器噪声配置

    用于模拟真实传感器的测量噪声，提高策略的鲁棒性（sim-to-real transfer）
    实际噪声 = level × scale_xxx × 随机值
    """
    level: float = 1.0                  # 噪声总开关/缩放因子，设为0可关闭所有噪声
    scale_joint_angle: float = 0.03     # 关节角度噪声标准差 [rad]
    scale_joint_vel: float = 1.5        # 关节速度噪声标准差 [rad/s]
    scale_gyro: float = 0.2             # 陀螺仪（角速度）噪声标准差 [rad/s]
    scale_gravity: float = 0.05         # 重力向量投影噪声标准差
    scale_linvel: float = 0.1           # 线速度噪声标准差 [m/s]


@dataclass
class ControlConfig:
    """
    控制参数配置

    注意：PD控制器的刚度(kp)和阻尼(kv)在XML文件中定义
    - kp = 200 N·m/rad（刚度，决定响应速度）
    - kv = 1 N·m·s/rad（阻尼，决定振荡抑制）
    """
    # 动作缩放系数：将神经网络输出[-1,1]映射到关节角度增量
    # 目标角度 = default_angles + action × action_scale
    # 0.06 rad ≈ 3.4°，保证动作平滑，避免剧烈抖动
    action_scale = 0.06


@dataclass
class InitState:
    """
    初始状态配置

    定义机器人在每个episode开始时的初始姿态
    """
    # 机器人在世界坐标系中的初始位置 [x, y, z]
    # z=0.5m 是站立时机身离地高度，与XML中模型定义匹配
    pos = [0.0, 0.0, 0.5]

    # 位置随机化范围 [x_min, y_min, x_max, y_max]
    # 机器人会在 20m × 20m 的区域内随机生成初始位置
    pos_randomization_range = [-10.0, -10.0, 10.0, 10.0]

    # 各关节的默认角度（站立姿态）
    # 关节命名规则：
    #   - LF/RF/LH/RH = 左前/右前/左后/右后
    #   - HAA = Hip Abduction/Adduction (髋关节外展/内收)
    #   - HFE = Hip Flexion/Extension (髋关节屈/伸)
    #   - KFE = Knee Flexion/Extension (膝关节屈/伸)
    default_joint_angles = {
        # 四个髋关节外展角度（控制腿的左右张开）
        "LF_HAA": 0.0,   # 左前髋外展 [rad]
        "RF_HAA": 0.0,   # 右前髋外展 [rad]
        "LH_HAA": 0.0,   # 左后髋外展 [rad]
        "RH_HAA": 0.0,   # 右后髋外展 [rad]
        # 四个髋关节屈伸角度（控制大腿前后摆动）
        # 前腿正值向前抬，后腿负值向后抬
        "LF_HFE": 0.4,   # 左前髋屈伸 [rad] ≈ 23°
        "RF_HFE": 0.4,   # 右前髋屈伸 [rad]
        "LH_HFE": -0.4,  # 左后髋屈伸 [rad]
        "RH_HFE": -0.4,  # 右后髋屈伸 [rad]
        # 四个膝关节角度（控制小腿弯曲）
        # 前腿负值向后弯，后腿正值向前弯（形成">"型站姿）
        "LF_KFE": -0.8,  # 左前膝关节 [rad] ≈ -46°
        "RF_KFE": -0.8,  # 右前膝关节 [rad]
        "LH_KFE": 0.8,   # 左后膝关节 [rad]
        "RH_KFE": 0.8,   # 右后膝关节 [rad]
    }


@dataclass
class Commands:
    """
    导航命令配置

    定义目标点生成的范围参数
    """
    # 目标位置相对于机器人初始位置的偏移范围
    # 格式: [dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max]
    # dx/dy: 位置偏移量 [米]，目标在机器人初始位置的±5m范围内
    # yaw: 目标朝向 [弧度]，在±π范围内随机（即任意朝向）
    pose_command_range = [-5.0, -5.0, -3.14, 5.0, 5.0, 3.14]


@dataclass
class Normalization:
    """
    观测值归一化系数

    将物理量归一化到合理范围，有助于神经网络训练
    归一化后的值 = 原始值 × 归一化系数
    """
    lin_vel = 2.0    # 线速度归一化系数（1 m/s → 2.0）
    ang_vel = 0.25   # 角速度归一化系数（1 rad/s → 0.25）
    dof_pos = 1.0    # 关节角度归一化系数（保持原值）
    dof_vel = 0.05   # 关节速度归一化系数（20 rad/s → 1.0）


@dataclass
class Asset:
    """
    机器人资产配置

    定义机器人模型中各部件的名称，用于查询物理状态
    """
    body_name = "base"  # 机身（base link）名称
    # 四只脚的碰撞体名称，用于检测足地接触
    foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
    # 触发终止条件的碰撞体列表（机身触地则终止episode）
    terminate_after_contacts_on = ["base"]
    ground_name = "ground"  # 地面碰撞体名称


@dataclass
class Sensor:
    """
    传感器名称映射

    对应 XML 文件中定义的传感器名称
    """
    base_linvel = "base_linvel"  # 机身线速度传感器（模拟IMU+状态估计）
    base_gyro = "base_gyro"      # 陀螺仪传感器（测量角速度）


@dataclass
class RewardConfig:
    """
    奖励函数权重配置

    注意：实际的奖励计算在 anymal_c_np.py 的 _compute_reward() 中
    这里的配置目前未被使用，实际权重硬编码在代码中
    """
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "termination": -400.0,         # 终止惩罚（摔倒等）
            "position_tracking": 0.5,      # 位置跟踪奖励
            "fine_position_tracking": 0.5, # 精细位置跟踪奖励
            "orientation": -0.2,           # 姿态偏离惩罚
        }
    )


@registry.envcfg("anymal_c_navigation_flat")  # 注册环境配置，可通过名称查找
@dataclass
class AnymalCEnvCfg(EnvCfg):
    """
    ANYmal C 导航环境主配置类

    整合所有子配置，并定义环境级别的参数
    """
    model_file: str = model_file          # MuJoCo 模型文件路径
    reset_noise_scale: float = 0.01       # 重置时的状态噪声缩放
    max_episode_seconds: float = 7.0      # 每个episode最长时间 [秒]
    sim_dt: float = 0.01                  # 仿真时间步长 [秒]（100Hz）
    ctrl_dt: float = 0.01                 # 控制时间步长 [秒]（与仿真同步）
    reset_yaw_scale: float = 0.1          # 重置时朝向随机化缩放
    max_dof_vel: float = 100.0            # 关节速度上限 [rad/s]，超过则终止

    # 子配置模块
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)

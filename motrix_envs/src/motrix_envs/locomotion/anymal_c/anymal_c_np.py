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
ANYmal C 四足机器人导航环境实现

本文件实现了 ANYmal C 四足机器人的平地导航任务环境，遵循 Gymnasium 接口规范。

任务描述：
    机器人从随机初始位置出发，需要移动到指定的目标位置，并调整到目标朝向后停稳。

核心流程：
    1. reset() - 重置环境，随机生成机器人位置和目标点
    2. apply_action() - 将神经网络输出转换为关节目标角度
    3. 物理仿真步进（由基类处理）
    4. update_state() - 提取观测、计算奖励、判断终止

观测空间 (54维)：
    - 机身线速度 (3) + 陀螺仪 (3) + 重力投影 (3)
    - 关节角度 (12) + 关节速度 (12) + 上一步动作 (12)
    - 速度命令 (3) + 位置误差 (2) + 朝向误差 (1)
    - 距离 (1) + 到达标志 (1) + 停稳标志 (1)

动作空间 (12维)：
    - 12个关节的位置增量，范围 [-1, 1]
    - 实际目标角度 = 默认角度 + 动作 × 0.06
"""

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.math.quaternion import Quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import AnymalCEnvCfg


@registry.env("anymal_c_navigation_flat", "np")  # 注册环境，名称为 "anymal_c_navigation_flat"，类型为 "np"
class AnymalCEnv(NpEnv):
    """
    ANYmal C 四足机器人导航环境

    继承自 NpEnv（基于 NumPy 的环境基类），实现了 Gymnasium 接口。

    主要职责：
    - 管理机器人状态和目标点
    - 将 RL 动作转换为关节控制
    - 构建观测向量
    - 计算奖励和终止条件
    """
    _cfg: AnymalCEnvCfg  # 类型提示：配置对象

    def __init__(self, cfg: AnymalCEnvCfg, num_envs: int = 1):
        """
        初始化环境

        Args:
            cfg: 环境配置对象，包含所有可调参数
            num_envs: 并行环境数量（用于向量化训练）
        """
        super().__init__(cfg, num_envs=num_envs)

        # 获取机身刚体对象，用于查询位姿和速度
        self._body = self._model.get_body(cfg.asset.body_name)
        # 初始化碰撞检测相关的几何体索引
        self._init_contact_geometry()

        # 获取目标标记物体（用于可视化目标位置）
        self._target_marker_body = self._model.get_body("target_marker")

        # ============ 定义动作空间和观测空间 ============
        # 动作空间：12个关节的位置增量，范围 [-1, 1]
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # 观测空间：54维连续向量
        # 组成：linvel(3) + gyro(3) + gravity(3) + joint_pos(12) + joint_vel(12) + last_actions(12) +
        #       commands(3) + position_error(2) + heading_error(1) + distance(1) + reached_flag(1) + stop_ready_flag(1)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)

        # ============ 记录模型的自由度信息 ============
        self._num_dof_pos = self._model.num_dof_pos  # 位置自由度数量（含浮动基座）
        self._num_dof_vel = self._model.num_dof_vel  # 速度自由度数量
        self._num_action = self._model.num_actuators  # 执行器（关节电机）数量 = 12

        # 初始关节位置和速度（从模型中读取）
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros(
            (self._model.num_dof_vel,),
            dtype=np.float32,
        )

        # 初始化内部缓冲区（默认关节角度、归一化系数等）
        self._init_buffer()

    def _init_buffer(self):
        """
        初始化内部缓冲区

        设置默认关节角度和归一化系数等运行时需要的常量
        """
        cfg = self._cfg

        # 默认关节角度数组（站立姿态时的关节位置）
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)

        # 速度命令的归一化系数 [vx_scale, vy_scale, vyaw_scale]
        # 用于将速度命令归一化到合理范围，便于神经网络处理
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel], dtype=np.float32
        )

        # 从配置中读取默认关节角度，按执行器顺序填充
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle

        # 将默认关节角度写入初始位置向量的最后12个元素（对应12个关节）
        # 前面的元素是浮动基座的位置和姿态
        self._init_dof_pos[-self._num_action :] = self.default_angles

    def _init_contact_geometry(self):
        """
        初始化碰撞检测所需的几何体索引

        建立碰撞检测矩阵，用于：
        1. 检测机身是否触地（触发终止条件）
        2. 检测足部是否接触地面（用于步态分析等）
        """
        cfg = self._cfg
        # 获取地面几何体的索引
        self.ground_index = self._model.get_geom_index(cfg.asset.ground_name)

        # 初始化终止条件碰撞检测（机身-地面）
        self._init_termination_contact()
        # 初始化足部碰撞检测（足部-地面）
        self._init_foot_contact()

    def _init_termination_contact(self):
        """
        初始化终止条件碰撞检测

        当机身（base）接触地面时，表示机器人摔倒，需要终止当前episode
        """
        cfg = self._cfg
        # 查找所有需要检测碰撞的机身几何体索引
        base_indices = []
        for base_name in cfg.asset.terminate_after_contacts_on:
            try:
                base_idx = self._model.get_geom_index(base_name)
                if base_idx is not None:
                    base_indices.append(base_idx)
                else:
                    print(f"Warning: Geom '{base_name}' not found in model")
            except Exception as e:
                print(f"Warning: Error finding base geom '{base_name}': {e}")

        # 创建碰撞检测矩阵：每行是一对需要检测碰撞的几何体 [机身几何体, 地面几何体]
        if base_indices:
            self.termination_contact = np.array([[idx, self.ground_index] for idx in base_indices], dtype=np.uint32)
            self.num_termination_check = self.termination_contact.shape[0]
        else:
            # 如果没有找到需要检测的几何体，使用空数组
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            print("Warning: No base contacts configured for termination")

    def _init_foot_contact(self):
        """
        初始化足部接触检测

        检测四只脚是否接触地面，可用于步态分析、奖励计算等
        """
        cfg = self._cfg
        foot_indices = []
        for foot_name in cfg.asset.foot_names:
            try:
                foot_idx = self._model.get_geom_index(foot_name)
                if foot_idx is not None:
                    foot_indices.append(foot_idx)
                else:
                    print(f"Warning: Foot geom '{foot_name}' not found in model")
            except Exception as e:
                print(f"Warning: Error finding foot geom '{foot_name}': {e}")

        # 创建足部-地面碰撞检测矩阵
        if foot_indices:
            self.foot_contact_check = np.array([[idx, self.ground_index] for idx in foot_indices], dtype=np.uint32)
            self.num_foot_check = self.foot_contact_check.shape[0]
        else:
            self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
            self.num_foot_check = 0
            print("Warning: No foot contacts configured")

    def get_dof_pos(self, data: mtx.SceneData):
        """获取关节位置（12个关节的角度）"""
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData):
        """获取关节速度（12个关节的角速度）"""
        return self._body.get_joint_dof_vel(data)

    def _extract_root_state(self, data):
        """
        提取机身（根节点）的状态信息

        Returns:
            root_pos: 机身位置 [num_envs, 3] - (x, y, z)
            root_quat: 机身姿态四元数 [num_envs, 4] - (x, y, z, w)
            root_linvel: 机身线速度 [num_envs, 3] - (vx, vy, vz)
        """
        # 获取机身位姿（位置+四元数）
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]      # 位置
        root_quat = pose[:, 3:7]    # 四元数姿态
        # 从传感器获取线速度（比直接计算更准确）
        root_linvel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_linvel

    @property
    def observation_space(self):
        """观测空间属性（Gymnasium 接口要求）"""
        return self._observation_space

    @property
    def action_space(self):
        """动作空间属性（Gymnasium 接口要求）"""
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        """
        应用动作到机器人

        将神经网络输出的动作转换为关节目标角度，并发送给 PD 控制器。

        Args:
            actions: 神经网络输出 [num_envs, 12]，范围 [-1, 1]
            state: 当前环境状态

        Returns:
            更新后的状态

        控制流程：
            1. 保存当前动作作为下一步的"上一步动作"
            2. 动作缩放：action × 0.06 → 关节角度增量（约±3.4°）
            3. 计算目标角度：默认角度 + 增量
            4. 发送给执行器（PD控制器会跟踪目标角度）
        """
        # 保存动作历史，用于计算动作变化率惩罚
        if "current_action" not in state.info:
            state.info["current_actions"] = np.zeros_like(actions)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions

        # 位置控制模式：计算目标关节角度
        # 目标角度 = 默认站立角度 + 动作增量
        actions_scaled = actions * self._cfg.control_config.action_scale
        state.data.actuator_ctrls = self.default_angles + actions_scaled
        return state

    def update_state(self, state: NpEnvState):
        """
        更新环境状态（核心方法）

        在物理仿真步进后调用，负责：
        1. 从仿真中提取机器人状态
        2. 计算导航相关信息（位置误差、速度命令等）
        3. 构建观测向量
        4. 计算奖励
        5. 判断终止条件
        6. 更新可视化标记

        Args:
            state: 包含仿真数据的状态对象

        Returns:
            更新后的状态（包含 obs, reward, terminated）
        """
        data = state.data

        # ============ 1. 提取机器人状态 ============
        # 获取机身位置、姿态、速度
        root_pos, root_quat, root_vel = self._extract_root_state(data)

        # 获取关节状态（12个腿部关节）
        joint_pos = self.get_dof_pos(data)  # 关节角度 [num_envs, 12]
        joint_vel = self.get_dof_vel(data)  # 关节速度 [num_envs, 12]
        joint_pos_rel = joint_pos - self.default_angles  # 相对于默认姿态的偏差

        # 获取传感器数据
        base_lin_vel = root_vel[:, :3]  # 机身线速度
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)  # 陀螺仪（角速度）
        projected_gravity = self._compute_projected_gravity(root_quat)  # 重力在机身坐标系的投影

        # ============ 2. 计算导航信息 ============
        # 获取目标位置和朝向
        pose_commands = state.info["pose_commands"]  # [num_envs, 3] = (x, y, yaw)
        robot_position = root_pos[:, :2]  # 机器人当前XY位置
        robot_heading = Quaternion.get_yaw(root_quat)  # 机器人当前朝向（yaw角）
        target_position = pose_commands[:, :2]  # 目标XY位置
        target_heading = pose_commands[:, 2]  # 目标朝向

        # 计算位置误差和距离
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        # 判断是否到达目标位置（阈值0.3米）
        position_threshold = 0.3
        reached_position = distance_to_target < position_threshold

        # ============ 3. 生成速度命令（P控制器） ============
        # 根据位置误差计算期望速度，用简单的P控制器
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)  # 增益1.0，限幅±1 m/s
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)  # 到达后停止

        # 计算朝向误差（处理角度环绕，确保在[-π, π]范围内）
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)

        # 判断是否到达目标朝向（阈值15°）
        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold

        # 综合判断：位置和朝向都到达
        reached_all = np.logical_and(reached_position, reached_heading)

        # 计算角速度命令（带死区，避免小幅抖动）
        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        deadband_yaw = np.deg2rad(8)  # 8°死区
        desired_yaw_rate = np.where(np.abs(heading_diff) < deadband_yaw, 0.0, desired_yaw_rate)

        # 到达后停止所有运动
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        state.info["desired_vel_xy"] = desired_vel_xy

        # 组合速度命令向量 [vx, vy, vyaw]
        velocity_commands = np.concatenate([desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1)

        # ============ 4. 归一化观测值 ============
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]

        # 任务相关观测的归一化
        position_error_normalized = position_error / 5.0  # 除以最大距离5m
        heading_error_normalized = heading_diff / np.pi   # 除以π，范围[-1, 1]
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)

        # 判断是否满足停稳条件：到达目标且角速度接近零
        stop_ready = np.logical_and(reached_all, np.abs(gyro[:, 2]) < 5e-2)
        stop_ready_flag = stop_ready.astype(np.float32)

        # ============ 5. 构建观测向量（54维） ============
        obs = np.concatenate(
            [
                noisy_linvel,                              # 3: 机身线速度
                noisy_gyro,                                # 3: 陀螺仪（角速度）
                projected_gravity,                         # 3: 重力投影（表示倾斜）
                noisy_joint_angle,                         # 12: 关节角度
                noisy_joint_vel,                           # 12: 关节速度
                last_actions,                              # 12: 上一步动作
                command_normalized,                        # 3: 速度命令
                position_error_normalized,                 # 2: 位置误差向量
                heading_error_normalized[:, np.newaxis],   # 1: 朝向误差
                distance_normalized[:, np.newaxis],        # 1: 到目标距离
                reached_flag[:, np.newaxis],               # 1: 是否到达
                stop_ready_flag[:, np.newaxis],            # 1: 是否停稳
            ],
            axis=-1,
        )
        assert obs.shape == (data.shape[0], 54)

        # ============ 6. 更新可视化标记 ============
        self._update_target_marker(data, pose_commands)  # 更新目标位置箭头
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)  # 更新方向箭头

        # ============ 7. 计算奖励和终止条件 ============
        reward = self._compute_reward(data, state.info, velocity_commands)
        terminated_state = self._compute_terminated(state)
        terminated = terminated_state.terminated

        # 更新状态
        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        return state

    def _update_heading_arrows(
        self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray
    ):
        """
        更新方向箭头可视化（仅用于调试/可视化，无物理效果）

        在仿真渲染中显示两个箭头：
        1. 绿色箭头：当前实际运动方向（基于机身线速度）
        2. 蓝色箭头：期望运动方向（基于速度命令）

        Args:
            data: 仿真数据
            robot_pos: 机器人位置 [num_envs, 3]
            desired_vel_xy: 期望XY速度 [num_envs, 2]
            base_lin_vel_xy: 实际XY速度 [num_envs, 2]
        """
        arrow_height = 0.76  # 箭头显示高度（机身高度0.56 + 偏移0.2）

        # 计算当前运动方向（从实际速度计算yaw角）
        # 只有速度大于阈值时才计算方向，避免静止时方向不稳定
        cur_yaw = np.where(
            np.linalg.norm(base_lin_vel_xy, axis=1) > 1e-3,
            np.arctan2(base_lin_vel_xy[:, 1], base_lin_vel_xy[:, 0]),
            0.0,
        )

        # 设置当前方向箭头位置和姿态
        robot_arrow_pos = robot_pos.copy()
        robot_arrow_pos[:, 2] = arrow_height
        robot_arrow_quat = Quaternion.from_euler(0, 0, cur_yaw)
        mocap = self._model.get_body("robot_heading_arrow").mocap
        mocap.set_pose(data, np.concatenate([robot_arrow_pos, robot_arrow_quat], axis=1))

        # 计算期望运动方向（从速度命令计算yaw角）
        des_yaw = np.where(
            np.linalg.norm(desired_vel_xy, axis=1) > 1e-6, np.arctan2(desired_vel_xy[:, 1], desired_vel_xy[:, 0]), 0.0
        )

        # 设置期望方向箭头位置和姿态
        desired_arrow_quat = Quaternion.from_euler(0, 0, des_yaw)
        mocap = self._model.get_body("desired_heading_arrow").mocap
        mocap.set_pose(data, np.concatenate([robot_arrow_pos, desired_arrow_quat], axis=1))

    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray) -> np.ndarray:
        """
        计算奖励函数（核心方法）

        奖励设计采用分阶段策略：
        - 到达前：鼓励跟踪速度命令、接近目标
        - 到达后：鼓励停稳、保持姿态

        Args:
            data: 仿真数据
            info: 状态信息字典（包含目标位置、历史动作等）
            velocity_commands: 速度命令 [num_envs, 3] = (vx, vy, vyaw)

        Returns:
            reward: 每个环境的奖励值 [num_envs]

        奖励组成：
        ┌─────────────────────────────────────────────────────────────┐
        │ 正向奖励：                                                   │
        │   - 线速度跟踪奖励（指数函数，误差越小奖励越高）              │
        │   - 角速度跟踪奖励（指数函数）                               │
        │   - 接近目标奖励（每接近1米奖励4分）                         │
        │   - 首次到达奖励（一次性10分）                               │
        │   - 停稳奖励（到达后保持静止）                               │
        ├─────────────────────────────────────────────────────────────┤
        │ 惩罚项：                                                     │
        │   - 终止惩罚（-20，摔倒等严重错误）                          │
        │   - Z轴线速度惩罚（抑制上下跳动）                            │
        │   - XY轴角速度惩罚（抑制翻滚和俯仰）                         │
        │   - 扭矩惩罚（节能）                                         │
        │   - 动作变化率惩罚（平滑控制）                               │
        └─────────────────────────────────────────────────────────────┘
        """
        # ============ 终止条件惩罚 ============
        termination_penalty = np.zeros(self._num_envs, dtype=np.float32)

        # 检查关节速度是否超限（防止仿真发散）
        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_overflow = vel_max > self._cfg.max_dof_vel  # 超过100 rad/s
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)
        termination_penalty = np.where(vel_overflow | vel_extreme, -20.0, termination_penalty)

        # 检查机身是否触地（摔倒）
        cquerys = self._model.get_contact_query(data)
        termination_check = cquerys.is_colliding(self.termination_contact)
        termination_check = termination_check.reshape((self._num_envs, self.num_termination_check))
        base_contact = termination_check.any(axis=1)
        termination_penalty = np.where(base_contact, -20.0, termination_penalty)

        # 检查是否侧翻（倾斜角超过75°）
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        proj_g = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)  # 重力在XY平面的分量
        gz = proj_g[:, 2]  # 重力在Z轴的分量
        tilt_angle = np.arctan2(gxy, np.abs(gz))  # 倾斜角
        side_flip_mask = tilt_angle > np.deg2rad(75)
        termination_penalty = np.where(side_flip_mask, -20.0, termination_penalty)

        # ============ 1. 线速度跟踪奖励 ============
        # 使用指数函数：误差为0时奖励为1，误差增大时奖励指数衰减
        base_lin_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        lin_vel_error = np.sum(np.square(velocity_commands[:, :2] - base_lin_vel[:, :2]), axis=1)
        tracking_lin_vel = np.exp(-lin_vel_error / 0.25)  # σ² = 0.25

        # ============ 2. 角速度跟踪奖励 ============
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        ang_vel_error = np.square(velocity_commands[:, 2] - gyro[:, 2])
        tracking_ang_vel = np.exp(-ang_vel_error / 0.25)

        # ============ 3. 到达判断和相关奖励 ============
        # 获取当前位置和目标位置
        robot_position = pose[:, :2]
        robot_heading = Quaternion.get_yaw(root_quat)
        target_position = info["pose_commands"][:, :2]
        target_heading = info["pose_commands"][:, 2]

        # 计算位置误差和距离
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        # 计算朝向误差
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)

        # 判断是否到达
        position_threshold = 0.3
        reached_position = distance_to_target < position_threshold
        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold
        reached_all = np.logical_and(reached_position, reached_heading)

        # 首次到达奖励（只给一次）
        info["ever_reached"] = info.get("ever_reached", np.zeros(self._num_envs, dtype=bool))
        first_time_reach = np.logical_and(reached_all, ~info["ever_reached"])
        info["ever_reached"] = np.logical_or(info["ever_reached"], reached_all)
        arrival_bonus = np.where(first_time_reach, 10.0, 0.0)

        # ============ 4. 接近目标奖励 ============
        # 基于历史最小距离计算进步量，鼓励持续接近目标
        if "min_distance" not in info:
            info["min_distance"] = distance_to_target.copy()
        distance_improvement = info["min_distance"] - distance_to_target  # 正值表示更近了
        info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)
        approach_reward = np.clip(distance_improvement * 4.0, -1.0, 1.0)  # 每接近1米奖励4分

        # ============ 5. 姿态稳定性惩罚 ============
        # 正常站立时重力投影应该是 [0, 0, -1]，偏离则惩罚
        projected_gravity = self._compute_projected_gravity(root_quat)
        orientation_penalty = (
            np.square(projected_gravity[:, 0])
            + np.square(projected_gravity[:, 1])
            + np.square(projected_gravity[:, 2] + 1.0)
        )

        # ============ 6. 停稳奖励（到达后） ============
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        zero_ang_mask = np.abs(gyro[:, 2]) < 0.05  # 角速度小于0.05 rad/s
        zero_ang_bonus = np.where(np.logical_and(reached_all, zero_ang_mask), 6.0, 0.0)
        # 停稳基础奖励：速度越小奖励越高
        stop_base = 2 * (0.8 * np.exp(-((speed_xy / 0.2) ** 2)) + 1.2 * np.exp(-((np.abs(gyro[:, 2]) / 0.1) ** 4)))
        stop_bonus = np.where(reached_all, stop_base + zero_ang_bonus, 0.0)

        # ============ 7. 各项惩罚 ============
        # Z轴线速度惩罚（抑制跳跃）
        lin_vel_z_penalty = np.square(base_lin_vel[:, 2])

        # XY轴角速度惩罚（抑制翻滚俯仰）
        ang_vel_xy_penalty = np.sum(np.square(gyro[:, :2]), axis=1)

        # 扭矩惩罚（节能）
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=1)

        # 关节速度惩罚
        joint_vel = self.get_dof_vel(data)
        dof_vel_penalty = np.sum(np.square(joint_vel), axis=1)

        # 动作变化率惩罚（平滑控制）
        action_diff = info["current_actions"] - info["last_actions"]
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)

        # ============ 8. 组合最终奖励 ============
        # 分两种情况：到达前和到达后
        reward = np.where(
            reached_all,
            # ===== 到达后：停稳为主 =====
            (
                stop_bonus                          # 停稳奖励
                + arrival_bonus                     # 首次到达奖励
                - 2.0 * lin_vel_z_penalty          # Z轴速度惩罚
                - 0.05 * ang_vel_xy_penalty        # XY角速度惩罚
                - 0.0 * orientation_penalty        # 姿态惩罚（当前关闭）
                - 0.00001 * torque_penalty         # 扭矩惩罚
                - 0.0 * dof_vel_penalty            # 关节速度惩罚（当前关闭）
                - 0.001 * action_rate_penalty      # 动作平滑惩罚
                + termination_penalty              # 终止惩罚
            ),
            # ===== 到达前：导航为主 =====
            (
                1.5 * tracking_lin_vel             # 线速度跟踪（主要）
                + 0.3 * tracking_ang_vel           # 角速度跟踪
                + approach_reward                  # 接近目标奖励
                - 2.0 * lin_vel_z_penalty          # Z轴速度惩罚
                - 0.05 * ang_vel_xy_penalty        # XY角速度惩罚
                - 0.0 * orientation_penalty        # 姿态惩罚（当前关闭）
                - 0.00001 * torque_penalty         # 扭矩惩罚
                - 0.0 * dof_vel_penalty            # 关节速度惩罚（当前关闭）
                - 0.001 * action_rate_penalty      # 动作平滑惩罚
                + termination_penalty              # 终止惩罚
            ),
        )

        return reward

    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """
        更新目标位置标记（绿色箭头）

        在仿真渲染中显示目标位置和朝向，便于调试和可视化。

        Args:
            data: 仿真数据
            pose_commands: 目标姿态命令 [num_envs, 3] = (x, y, yaw)
        """
        num_envs = data.shape[0]

        # 设置箭头位置（在目标位置上方0.5米处）
        arrow_pos = pose_commands.copy()
        arrow_pos[:, 2] = 0.05  # 这行被下面覆盖了，可能是历史遗留
        arrow_pos = np.column_stack([pose_commands[:, 0], pose_commands[:, 1], np.full((num_envs, 1), 0.5)])

        # 根据目标朝向计算箭头姿态四元数
        arrow_quat = Quaternion.from_euler(0, 0, pose_commands[:, 2])

        # 更新 mocap body 的位姿（mocap 是运动捕捉物体，不参与物理仿真）
        mocap = self._model.get_body("target_marker").mocap
        mocap.set_pose(data, np.concatenate([arrow_pos, arrow_quat], axis=1))

    def _compute_terminated(self, state: NpEnvState) -> NpEnvState:
        """
        计算终止条件

        检查是否满足任意终止条件，满足则结束当前episode。

        终止条件包括：
        1. 关节速度超限（>100 rad/s）- 防止仿真发散
        2. 数值异常（NaN/Inf）- 仿真崩溃
        3. 机身触地 - 摔倒
        4. 倾斜角过大（>75°）- 即将摔倒

        Args:
            state: 当前环境状态

        Returns:
            更新了 terminated 标志的状态
        """
        data = state.data
        terminated = np.zeros(self._num_envs, dtype=bool)

        # ============ 1. 关节速度检查 ============
        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)
        # 速度超过阈值（训练初期设置较宽松的100 rad/s）
        vel_overflow = vel_max > self._cfg.max_dof_vel
        # 极端情况：NaN、Inf 或超大值
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)
        terminated = np.logical_or(terminated, vel_overflow)
        terminated = np.logical_or(terminated, vel_extreme)

        # ============ 2. 机身触地检查 ============
        cquerys = self._model.get_contact_query(data)
        termination_check = cquerys.is_colliding(self.termination_contact)
        termination_check = termination_check.reshape((self._num_envs, self.num_termination_check))
        base_contact = termination_check.any(axis=1)  # 任意一个机身几何体触地
        terminated = np.logical_or(terminated, base_contact)

        # ============ 3. 侧翻检查 ============
        # 通过重力投影判断倾斜角度
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        proj_g = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)  # 重力在机身XY平面的分量
        gz = proj_g[:, 2]  # 重力在机身Z轴的分量
        tilt_angle = np.arctan2(gxy, np.abs(gz))  # 计算倾斜角
        side_flip_mask = tilt_angle > np.deg2rad(75)  # 超过75°视为侧翻
        terminated = np.logical_or(terminated, side_flip_mask)

        return state.replace(terminated=terminated)

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """
        重置环境（核心方法）

        在每个episode开始时调用，负责：
        1. 随机生成机器人初始位置
        2. 随机生成目标位置和朝向
        3. 初始化机器人姿态（站立状态）
        4. 构建初始观测向量
        5. 初始化状态信息字典

        Args:
            data: 仿真数据对象
            done: 需要重置的环境掩码（可选，用于向量化环境部分重置）

        Returns:
            obs: 初始观测向量 [num_envs, 54]
            info: 状态信息字典，包含：
                - pose_commands: 目标姿态 [num_envs, 3]
                - last_actions: 上一步动作
                - current_actions: 当前动作
                - ever_reached: 是否曾到达目标
                - min_distance: 历史最小距离
        """
        cfg: AnymalCEnvCfg = self._cfg
        num_envs = data.shape[0]

        # ============ 1. 生成机器人初始位置 ============
        # 在 20m × 20m 区域内随机分布
        pos_range = cfg.init_state.pos_randomization_range
        robot_init_x = np.random.uniform(
            pos_range[0],   # x_min = -10
            pos_range[2],   # x_max = 10
            num_envs,
        )
        robot_init_y = np.random.uniform(
            pos_range[1],   # y_min = -10
            pos_range[3],   # y_max = 10
            num_envs,
        )
        robot_init_pos = np.stack([robot_init_x, robot_init_y], axis=1)  # [num_envs, 2]

        # ============ 2. 生成目标位置和朝向 ============
        # 目标位置 = 机器人初始位置 + 随机偏移（±5m）
        target_offset = np.random.uniform(
            low=cfg.commands.pose_command_range[:2],   # [dx_min, dy_min] = [-5, -5]
            high=cfg.commands.pose_command_range[3:5], # [dx_max, dy_max] = [5, 5]
            size=(num_envs, 2)
        )
        target_positions = robot_init_pos + target_offset  # 目标在世界坐标系中的位置

        # 目标朝向：在 [-π, π] 范围内随机
        target_headings = np.random.uniform(
            low=cfg.commands.pose_command_range[2],    # yaw_min = -π
            high=cfg.commands.pose_command_range[5],   # yaw_max = π
            size=(num_envs, 1)
        )

        # 组合目标姿态命令 [x, y, yaw]
        pose_commands = np.concatenate([target_positions, target_headings], axis=1)

        # ============ 3. 设置机器人初始状态 ============
        # 复制默认的自由度位置和速度
        init_dof_pos = np.tile(self._init_dof_pos, (*data.shape, 1))
        init_dof_vel = np.tile(self._init_dof_vel, (*data.shape, 1))

        # 创建位置偏移（只修改XY位置，不动姿态四元数）
        noise_pos = np.zeros((*data.shape, self._num_dof_pos), dtype=np.float32)

        # 设置机身初始XY位置（相对于默认位置的偏移）
        noise_pos[:, 0] = robot_init_x - cfg.init_state.pos[0]  # X偏移
        noise_pos[:, 1] = robot_init_y - cfg.init_state.pos[1]  # Y偏移
        # Z轴不加噪声，保持固定高度0.5m

        # 所有速度初始化为0（完全静止）
        noise_vel = np.zeros((*data.shape, self._num_dof_vel), dtype=np.float32)

        # 应用初始状态
        dof_pos = init_dof_pos + noise_pos
        dof_vel = init_dof_vel + noise_vel

        # 重置仿真数据并设置状态
        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)  # 更新正运动学（计算各部件位姿）

        # 更新目标位置可视化标记
        self._update_target_marker(data, pose_commands)

        # ============ 4. 提取初始状态并构建观测 ============
        # 获取机身状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)

        # 获取关节状态
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles  # 相对于默认姿态的偏差

        # 获取传感器数据
        base_lin_vel = root_vel[:, :3]
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # 计算速度命令（与 update_state 保持一致）
        robot_position = root_pos[:, :2]
        robot_heading = Quaternion.get_yaw(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]

        # 计算位置误差和距离
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        # 判断是否到达（初始时通常不会到达，阈值0.1米）
        position_threshold = 0.1
        reached_position = distance_to_target < position_threshold

        # 计算期望XY速度
        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)  # 到达后速度为0

        # 实际XY线速度（用于可视化）
        base_lin_vel_xy = base_lin_vel[:, :2]

        # 更新方向箭头可视化（无物理效果）
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)

        # 计算朝向误差（处理角度环绕）
        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)

        # 判断是否到达目标朝向（阈值15°）
        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold

        # 计算期望角速度
        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        reached_all = np.logical_and(reached_position, reached_heading)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)  # 到达后速度为0
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)  # 到达后速度为0

        # 确保 desired_yaw_rate 是1D数组
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()

        # 组合速度命令向量
        velocity_commands = np.concatenate([desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1)

        # ============ 5. 归一化观测值（与 update_state 保持一致） ============
        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)  # 初始无动作历史

        # 任务相关观测归一化
        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)

        # 判断是否满足停稳条件
        stop_ready = np.logical_and(reached_all, np.abs(gyro[:, 2]) < 5e-2)
        stop_ready_flag = stop_ready.astype(np.float32)

        # ============ 6. 构建观测向量（54维） ============
        obs = np.concatenate(
            [
                noisy_linvel,                              # 3: 机身线速度
                noisy_gyro,                                # 3: 陀螺仪
                projected_gravity,                         # 3: 重力投影
                noisy_joint_angle,                         # 12: 关节角度
                noisy_joint_vel,                           # 12: 关节速度
                last_actions,                              # 12: 上一步动作（初始为0）
                command_normalized,                        # 3: 速度命令
                position_error_normalized,                 # 2: 位置误差
                heading_error_normalized[:, np.newaxis],   # 1: 朝向误差
                distance_normalized[:, np.newaxis],        # 1: 距离
                reached_flag[:, np.newaxis],               # 1: 是否到达
                stop_ready_flag[:, np.newaxis],            # 1: 是否停稳
            ],
            axis=-1,
        )
        assert obs.shape == (num_envs, 54)

        # ============ 7. 初始化状态信息字典 ============
        info = {
            "pose_commands": pose_commands,                                           # 目标姿态 [x, y, yaw]
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32), # 上一步动作
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),  # 当前动作
            "ever_reached": np.zeros(num_envs, dtype=bool),                           # 是否曾到达目标
            "min_distance": distance_to_target.copy(),                                # 历史最小距离（用于接近奖励）
        }

        return obs, info

    def _compute_projected_gravity(self, quat: np.ndarray) -> np.ndarray:
        """
        计算重力在机身坐标系中的投影

        将世界坐标系的重力向量 [0, 0, -1] 转换到机身坐标系。
        用于反映机身的倾斜状态：
        - 正常站立时：投影 ≈ [0, 0, -1]
        - 向前倾斜时：投影的X分量为负
        - 向左倾斜时：投影的Y分量为正

        Args:
            quat: 机身姿态四元数 [num_envs, 4]

        Returns:
            projected_gravity: 重力投影向量 [num_envs, 3]
        """
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return Quaternion.rotate_vector(quat, gravity)

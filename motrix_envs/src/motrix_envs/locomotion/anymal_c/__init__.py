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
ANYmal C 四足机器人导航环境模块

本模块实现了基于 MuJoCo 物理引擎的 ANYmal C 四足机器人平地导航任务。

主要导出：
- AnymalCEnv: 环境类，符合 Gymnasium 接口规范
- AnymalCEnvCfg: 环境配置类，包含所有可调参数

子模块：
- anymal_c_np: 基于 NumPy 的环境实现
- cfg: 配置定义

使用示例：
    from motrix_envs.locomotion.anymal_c import AnymalCEnv, AnymalCEnvCfg

    cfg = AnymalCEnvCfg()
    env = AnymalCEnv(cfg, num_envs=1)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from . import anymal_c_np, cfg  # noqa: F401  # 导入子模块
from .anymal_c_np import AnymalCEnv  # noqa: F401  # 环境类
from .cfg import AnymalCEnvCfg  # noqa: F401  # 配置类

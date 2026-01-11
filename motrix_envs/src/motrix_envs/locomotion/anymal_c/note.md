# ANYmal C 代码阅读笔记

本文档整理了阅读 ANYmal C 四足机器人导航环境代码时涉及的知识点。

---

## 目录

1. [Python 类与面向对象](#1-python-类与面向对象)
2. [类型注解（Type Hints）](#2-类型注解type-hints)
3. [装饰器（Decorators）](#3-装饰器decorators)
4. [NumPy 数组操作](#4-numpy-数组操作)
5. [四元数与姿态表示](#5-四元数与姿态表示)
6. [不可变数据结构](#6-不可变数据结构)
7. [强化学习环境概念](#7-强化学习环境概念)
8. [注册表模式（Registry Pattern）](#8-注册表模式registry-pattern)

---

## 1. Python 类与面向对象

### 1.1 `self` 关键字

`self` 是 Python 类中方法的第一个参数，代表类的实例对象本身。

```python
class AnymalCEnv:
    def __init__(self, cfg, num_envs=1):
        self._cfg = cfg  # self._cfg 是这个实例的属性
        self._num_envs = num_envs

    def get_dof_pos(self, data):
        # self 让方法可以访问实例的属性
        return self._body.get_joint_dof_pos(data)
```

**要点**：
- `self` 不是 Python 关键字，只是约定俗成的名称
- 通过 `self.xxx` 访问实例属性
- 通过 `self.method()` 调用实例方法
- 类似于其他语言中的 `this`

### 1.2 下划线命名规则

Python 使用下划线来表示访问级别（约定，非强制）：

| 命名方式 | 含义 | 示例 |
|---------|------|------|
| `name` | 公开属性/方法 | `observation_space` |
| `_name` | 内部使用（protected） | `_cfg`, `_body`, `_init_buffer()` |
| `__name` | 私有（name mangling） | `__private_var` |
| `__name__` | 魔术方法/特殊属性 | `__init__`, `__len__` |

```python
class AnymalCEnv:
    _cfg: AnymalCEnvCfg          # 内部配置，不建议外部直接访问
    _num_envs: int               # 内部状态

    @property
    def observation_space(self):  # 公开接口
        return self._observation_space
```

### 1.3 `super()` 函数

用于调用父类的方法，常见于继承场景：

```python
class AnymalCEnv(NpEnv):
    def __init__(self, cfg, num_envs=1):
        super().__init__(cfg, num_envs=num_envs)  # 调用父类 NpEnv 的 __init__
        # 子类特有的初始化...
```

---

## 2. 类型注解（Type Hints）

### 2.1 基本语法

Python 3.5+ 支持类型注解，用于提示变量/参数/返回值的类型：

```python
# 变量类型注解
_cfg: AnymalCEnvCfg  # 声明 _cfg 是 AnymalCEnvCfg 类型

# 函数参数和返回值类型注解
def __init__(self, cfg: AnymalCEnvCfg, num_envs: int = 1):
    pass

def get_dof_pos(self, data: mtx.SceneData) -> np.ndarray:
    return self._body.get_joint_dof_pos(data)
```

### 2.2 类属性类型注解

在类体中直接写类型注解（不赋值）是声明实例属性的类型：

```python
class AnymalCEnv(NpEnv):
    _cfg: AnymalCEnvCfg  # 只是类型提示，不创建属性

    def __init__(self, cfg):
        self._cfg = cfg  # 真正创建属性是在这里
```

**注意**：
- 类型注解是给开发工具（IDE、类型检查器）看的
- Python 运行时不会强制检查类型
- 可以使用 `mypy` 等工具进行静态类型检查

### 2.3 常用类型

```python
from typing import List, Dict, Tuple, Optional, Union

# 基本类型
x: int = 1
y: float = 1.0
s: str = "hello"
b: bool = True

# 容器类型
nums: List[int] = [1, 2, 3]
data: Dict[str, float] = {"a": 1.0}
point: Tuple[float, float] = (1.0, 2.0)

# 可选类型（可能为 None）
value: Optional[int] = None  # 等价于 Union[int, None]

# NumPy 数组
arr: np.ndarray = np.zeros(10)
```

---

## 3. 装饰器（Decorators）

### 3.1 什么是装饰器

装饰器是一种用于修改函数或类行为的语法糖，本质上是一个接收函数/类并返回新函数/类的高阶函数。

```python
@decorator
def func():
    pass

# 等价于
def func():
    pass
func = decorator(func)
```

### 3.2 `@property` 装饰器

将方法变成属性访问器，实现只读属性或计算属性：

```python
class AnymalCEnv:
    def __init__(self):
        self._observation_space = gym.spaces.Box(...)

    @property
    def observation_space(self):
        """可以像属性一样访问：env.observation_space"""
        return self._observation_space

# 使用
env = AnymalCEnv()
space = env.observation_space  # 不需要加括号，像访问属性一样
```

**优势**：
- 提供只读属性（没有 setter 就无法修改）
- 可以添加验证逻辑
- 接口统一（外部看起来像属性）

### 3.3 `@dataclass` 装饰器

Python 3.7+ 提供，自动生成 `__init__`、`__repr__`、`__eq__` 等方法：

```python
from dataclasses import dataclass, field

@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5

# 自动生成的 __init__ 等价于：
# def __init__(self, level=1.0, scale_joint_angle=0.03, scale_joint_vel=1.5):
#     self.level = level
#     self.scale_joint_angle = scale_joint_angle
#     self.scale_joint_vel = scale_joint_vel
```

使用 `field()` 处理可变默认值：

```python
@dataclass
class RewardConfig:
    # 错误：scales: dict = {}  # 可变对象作为默认值会被共享！
    # 正确：使用 field(default_factory=...)
    scales: dict = field(default_factory=lambda: {"termination": -400.0})
```

### 3.4 `@staticmethod` 和 `@classmethod`

```python
class MyClass:
    class_var = 10

    @staticmethod
    def static_func(x):
        """静态方法：不接收 self 或 cls，与实例无关"""
        return x * 2

    @classmethod
    def class_func(cls, x):
        """类方法：接收 cls（类本身），可以访问类属性"""
        return x + cls.class_var

# 使用
MyClass.static_func(5)   # 10
MyClass.class_func(5)    # 15
```

### 3.5 注册装饰器（Registry Pattern）

自定义装饰器，用于将类/函数注册到全局字典：

```python
@registry.env("anymal_c_navigation_flat", "np")
class AnymalCEnv(NpEnv):
    pass
```

详见 [第8节：注册表模式](#8-注册表模式registry-pattern)。

---

## 4. NumPy 数组操作

### 4.1 数组切片语法

NumPy 使用 `[row, col]` 语法进行多维数组索引：

```python
pose = body.get_pose(data)  # shape: [num_envs, 7]

# 基本切片
root_pos = pose[:, :3]      # 所有行，前3列 → [num_envs, 3]
root_quat = pose[:, 3:7]    # 所有行，第3-6列 → [num_envs, 4]
```

**切片语法详解**：

| 语法 | 含义 | 示例 |
|------|------|------|
| `:` | 所有元素 | `arr[:]` 取全部 |
| `:n` | 前n个 | `arr[:3]` 取前3个 |
| `n:` | 从第n个开始 | `arr[3:]` 从第3个到末尾 |
| `m:n` | 从第m到第n-1个 | `arr[3:7]` 取第3-6个 |
| `-n:` | 最后n个 | `arr[-3:]` 取最后3个 |

**二维数组切片**：

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

arr[:, 0]      # 所有行的第0列 → [1, 5, 9]
arr[0, :]      # 第0行的所有列 → [1, 2, 3, 4]
arr[:, :2]     # 所有行的前2列 → [[1,2], [5,6], [9,10]]
arr[1:, 2:]    # 第1行开始，第2列开始 → [[7,8], [11,12]]
```

### 4.2 `np.newaxis` 增加维度

用于在特定位置增加一个维度（大小为1）：

```python
arr = np.array([1, 2, 3])  # shape: (3,)

arr[:, np.newaxis]  # shape: (3, 1) → [[1], [2], [3]]
arr[np.newaxis, :]  # shape: (1, 3) → [[1, 2, 3]]
```

常见用途：广播运算时对齐维度

```python
heading_error_normalized = heading_diff / np.pi  # shape: (num_envs,)

# 拼接时需要变成 (num_envs, 1) 才能与其他 2D 数组拼接
obs = np.concatenate([
    ...,
    heading_error_normalized[:, np.newaxis],  # (num_envs,) → (num_envs, 1)
], axis=-1)
```

### 4.3 `np.where` 条件选择

根据条件选择不同的值：

```python
# np.where(condition, x, y)
# condition 为 True 时选 x，否则选 y

desired_vel_xy = np.where(
    reached_position[:, np.newaxis],  # 条件
    0.0,                               # True 时的值
    desired_vel_xy                     # False 时的值
)
```

### 4.4 常用数组操作

```python
# 创建数组
np.zeros((3, 4))           # 全0数组
np.ones((3, 4))            # 全1数组
np.full((3, 4), 5.0)       # 填充指定值
np.random.uniform(-1, 1, (3, 4))  # 均匀分布随机数

# 数组变形
arr.reshape((2, 6))        # 改变形状
arr.flatten()              # 展平为1D
np.tile(arr, (3, 1))       # 重复平铺

# 拼接
np.concatenate([a, b], axis=0)  # 按行拼接
np.concatenate([a, b], axis=1)  # 按列拼接
np.stack([a, b], axis=0)        # 新增维度堆叠
np.column_stack([a, b])         # 按列堆叠

# 数学运算
np.sum(arr, axis=1)        # 按行求和
np.mean(arr, axis=0)       # 按列求均值
np.linalg.norm(arr, axis=1)  # 按行求范数
np.clip(arr, -1, 1)        # 限制范围
np.square(arr)             # 平方
np.exp(arr)                # 指数
```

---

## 5. 四元数与姿态表示

### 5.1 为什么用四元数

三维空间中表示姿态（旋转）有多种方式：

| 表示方式 | 优点 | 缺点 |
|---------|------|------|
| 欧拉角 (roll, pitch, yaw) | 直观易理解 | 万向锁问题、插值不连续 |
| 旋转矩阵 (3×3) | 无万向锁 | 9个参数，冗余，正交约束难维护 |
| **四元数** (x, y, z, w) | 无万向锁、4参数、插值平滑 | 不够直观 |

### 5.2 四元数基本概念

四元数 `q = (x, y, z, w)` 可以表示绑某个轴旋转一定角度：

```
q = (sin(θ/2) * axis_x, sin(θ/2) * axis_y, sin(θ/2) * axis_z, cos(θ/2))
```

其中：
- `(axis_x, axis_y, axis_z)` 是旋转轴的单位向量
- `θ` 是旋转角度

**性质**：
- 单位四元数满足 `x² + y² + z² + w² = 1`
- `w = 1, x = y = z = 0` 表示无旋转（单位元）
- 四元数乘法表示旋转的组合

### 5.3 代码中的四元数操作

```python
from motrix_envs.math.quaternion import Quaternion

# 从机身位姿中提取四元数
pose = body.get_pose(data)  # [num_envs, 7] = [x, y, z, qx, qy, qz, qw]
root_quat = pose[:, 3:7]    # 四元数部分

# 从四元数提取 yaw 角（绑 Z 轴的旋转）
robot_heading = Quaternion.get_yaw(root_quat)  # 返回弧度值

# 从欧拉角创建四元数
quat = Quaternion.from_euler(roll, pitch, yaw)

# 用四元数旋转向量
gravity = np.array([0.0, 0.0, -1.0])
projected_gravity = Quaternion.rotate_vector(root_quat, gravity)
```

### 5.4 重力投影的物理意义

```python
def _compute_projected_gravity(self, quat):
    """将世界坐标系的重力转换到机身坐标系"""
    gravity = np.array([0.0, 0.0, -1.0])  # 世界坐标系中重力向下
    return Quaternion.rotate_vector(quat, gravity)
```

**解释**：
- 机器人正常站立时，机身坐标系与世界坐标系对齐
- 此时重力投影 ≈ `[0, 0, -1]`（重力指向机身的 -Z 方向）
- 当机器人前倾时，投影的 X 分量变负（重力相对机身有向后的分量）
- 当机器人左倾时，投影的 Y 分量变正

这个投影向量反映了机身的倾斜状态，是观测向量的重要组成部分。

---

## 6. 不可变数据结构

### 6.1 `state.replace()` 方法

在某些设计中，状态对象是不可变的（immutable）。修改状态需要创建新副本：

```python
def _compute_terminated(self, state: NpEnvState) -> NpEnvState:
    terminated = ...  # 计算终止条件

    # 不能直接修改：state.terminated = terminated  # 错误！
    # 而是创建一个新的状态对象，只更新 terminated 字段
    return state.replace(terminated=terminated)
```

### 6.2 为什么使用不可变数据

**优点**：
1. **线程安全**：多线程环境下无需加锁
2. **可追溯**：历史状态不会被意外修改
3. **函数式编程**：更容易推理代码行为
4. **缓存友好**：可以安全地缓存状态

**常见实现**：
- Python 的 `namedtuple`
- `dataclasses.dataclass(frozen=True)`
- 自定义的 `replace()` 方法

```python
from dataclasses import dataclass

@dataclass(frozen=True)  # frozen=True 使其不可变
class Point:
    x: float
    y: float

p1 = Point(1.0, 2.0)
# p1.x = 3.0  # 错误！无法修改

# 需要使用 replace 创建新对象
from dataclasses import replace
p2 = replace(p1, x=3.0)  # p2 = Point(3.0, 2.0)
```

---

## 7. 强化学习环境概念

### 7.1 Gymnasium 接口

标准的 RL 环境接口（原 OpenAI Gym）：

```python
import gymnasium as gym

# 核心方法
obs, info = env.reset()                    # 重置环境
obs, reward, terminated, truncated, info = env.step(action)  # 执行动作

# 核心属性
env.observation_space  # 观测空间定义
env.action_space       # 动作空间定义
```

### 7.2 观测空间与动作空间

使用 `gym.spaces` 定义：

```python
# 连续空间：Box
action_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(12,),      # 12维动作
    dtype=np.float32
)

observation_space = gym.spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(54,),      # 54维观测
    dtype=np.float32
)

# 离散空间
action_space = gym.spaces.Discrete(4)  # 0, 1, 2, 3 四个动作
```

### 7.3 ANYmal C 环境的具体定义

**观测空间（54维）**：

| 维度 | 内容 | 说明 |
|------|------|------|
| 0-2 | 机身线速度 | vx, vy, vz |
| 3-5 | 陀螺仪 | 角速度 ωx, ωy, ωz |
| 6-8 | 重力投影 | 机身倾斜状态 |
| 9-20 | 关节角度 | 12个关节的位置 |
| 21-32 | 关节速度 | 12个关节的速度 |
| 33-44 | 上一步动作 | 用于平滑控制 |
| 45-47 | 速度命令 | vx_cmd, vy_cmd, ωz_cmd |
| 48-49 | 位置误差 | dx, dy 到目标的距离 |
| 50 | 朝向误差 | 角度差 |
| 51 | 距离 | 到目标的欧氏距离 |
| 52 | 到达标志 | 是否到达目标 |
| 53 | 停稳标志 | 是否已停稳 |

**动作空间（12维）**：

12个关节的位置增量，范围 `[-1, 1]`

```
目标角度 = 默认站立角度 + 动作 × 0.06
```

### 7.4 DOF（自由度）

**DOF = Degrees of Freedom（自由度）**

- 描述系统可以独立运动的方式数量
- ANYmal C 有 12 个关节自由度（4条腿 × 3关节/腿）

关节命名规则：
- `LF/RF/LH/RH` = 左前/右前/左后/右后
- `HAA` = Hip Abduction/Adduction（髋关节外展/内收）
- `HFE` = Hip Flexion/Extension（髋关节屈/伸）
- `KFE` = Knee Flexion/Extension（膝关节屈/伸）

```
前视图：
    LF ---- RF
     |      |
    LH ---- RH

每条腿的关节：
    HAA (外展) → HFE (髋屈) → KFE (膝屈)
```

### 7.5 Episode 和终止条件

**Episode**：一次完整的训练/测试回合

**终止条件（terminated）**：
1. 关节速度超限（>100 rad/s）
2. 数值异常（NaN/Inf）
3. 机身触地（摔倒）
4. 倾斜角过大（>75°）

**截断条件（truncated）**：
- 时间超限（>7秒）

---

## 8. 注册表模式（Registry Pattern）

### 8.1 核心思想

将类/函数注册到一个全局字典中，通过名称字符串查找和创建实例。

```python
# 简化的注册表实现
class Registry:
    def __init__(self):
        self._envs = {}
        self._cfgs = {}

    def env(self, name, backend):
        """装饰器：注册环境类"""
        def decorator(cls):
            key = (name, backend)
            self._envs[key] = cls
            return cls  # 返回原类，不做修改
        return decorator

    def get_env(self, name, backend):
        """根据名称获取环境类"""
        return self._envs[(name, backend)]

registry = Registry()  # 全局单例
```

### 8.2 使用方式

**注册环境**：

```python
@registry.env("anymal_c_navigation_flat", "np")
class AnymalCEnv(NpEnv):
    pass

# 等价于：
class AnymalCEnv(NpEnv):
    pass
AnymalCEnv = registry.env("anymal_c_navigation_flat", "np")(AnymalCEnv)
# 实际效果：registry._envs[("anymal_c_navigation_flat", "np")] = AnymalCEnv
```

**查找并创建环境**：

```python
# 训练脚本中
env_cls = registry.get_env("anymal_c_navigation_flat", "np")
cfg = registry.get_cfg("anymal_c_navigation_flat")
env = env_cls(cfg, num_envs=1024)
```

### 8.3 优势

1. **解耦**：训练代码不需要直接 import 具体的环境类
2. **配置驱动**：通过配置文件指定环境名称即可切换
3. **可扩展**：添加新环境只需写装饰器，无需修改其他代码
4. **自动发现**：import 模块时自动注册

### 8.4 完整流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. 定义阶段                                                 │
│  @registry.env("anymal_c_navigation_flat", "np")            │
│  class AnymalCEnv: ...                                      │
│        ↓                                                    │
│  registry._envs[("anymal_c_navigation_flat", "np")] = cls   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 使用阶段（训练脚本）                                     │
│  env_cls = registry.get_env("anymal_c_navigation_flat", "np")│
│  env = env_cls(cfg, num_envs=1024)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 附录：代码架构总结

```
anymal_c/
├── __init__.py          # 模块入口，导出 AnymalCEnv, AnymalCEnvCfg
├── cfg.py               # 配置类定义
│   ├── NoiseConfig      # 传感器噪声
│   ├── ControlConfig    # 控制参数 (action_scale=0.06)
│   ├── InitState        # 初始状态 (default_joint_angles)
│   ├── Commands         # 导航命令范围
│   ├── Normalization    # 观测归一化系数
│   ├── Asset            # 机器人部件名称
│   ├── Sensor           # 传感器名称
│   ├── RewardConfig     # 奖励权重
│   └── AnymalCEnvCfg    # 主配置类（整合所有子配置）
├── anymal_c_np.py       # 环境实现
│   └── AnymalCEnv       # 环境类
│       ├── __init__()           # 初始化
│       ├── reset()              # 重置环境
│       ├── apply_action()       # 应用动作
│       ├── update_state()       # 更新状态
│       ├── _compute_reward()    # 计算奖励
│       └── _compute_terminated()# 计算终止条件
└── xmls/
    └── scene.xml        # MuJoCo 模型文件
```

**数据流**：

```
reset() → 生成初始状态和目标点
    ↓
apply_action() → 动作(12维) → 关节目标角度
    ↓
物理仿真步进 (基类处理)
    ↓
update_state() → 提取状态 → 构建观测(54维) → 计算奖励 → 判断终止
    ↓
返回 (obs, reward, terminated, truncated, info)
```

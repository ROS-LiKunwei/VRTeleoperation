# PICO4 手势遥操作全流程数据流文档

## 1. 系统概述

本系统实现了从PICO4 VR头显手势识别到SYSMO-32双臂机器人末端执行器控制的完整遥操作链路。用户在PICO4头显中做出手势，系统实时捕捉手腕位姿和手部26个关节的3D坐标，经过坐标变换、运动重定向等处理，最终驱动SYSMO-32双臂（每臂6自由度）末端执行器运动。

**目标机器人**：SYSMO-32（6-DOF双臂，共12个旋转关节）

## 2. 数据流全景图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PICO4 VR头显 (Unity端)                           │
│  GestureDetectorXR.cs → NetMQController.cs → ZMQ PUSH (8087/8110)      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ ZMQ TCP (手部关键点数据)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        BeaVR-bot (Python端)                             │
│                                                                         │
│  [1] PICO4VRHandDetector (pico4.py)                                    │
│      │ ZMQ PULL (8087/8110) → 解析 → ZMQ PUB (8088)                   │
│      ▼                                                                  │
│  [2] TransformHandPositionCoords (keypoint_transform.py)               │
│      │ ZMQ SUB (8088) → 坐标变换+平滑 → ZMQ PUB (8092/8093)           │
│      ▼                                                                  │
│  [3] Sysmo32Operator (sysmo32_operator.py)                             │
│      │ ZMQ SUB (8092/8093) → 运动重定向 → ZMQ PUB (10011/10013)       │
│      ▼                                                                  │
│  [4] Sysmo32Robot (sysmo32_robot.py)                                   │
│      │ ZMQ SUB (10011/10013) → 机器人控制 → MockSysmo32Control        │
│      ▼                                                                  │
│  [5] MuJoCo仿真 (mujoco_sim.py)                                       │
│      ZMQ SUB (10011/10013) → 逆运动学 → MuJoCo仿真渲染                │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3. 各环节数据格式详解

### 3.1 环节一：PICO4数据获取（Unity端）

**文件**：`BeaVR-app/BeaVR-Unity/Assets/Scripts/Gesture Detection/GestureDetectorXR.cs`

**功能**：从PICO4 XR手部追踪系统获取26个关节的3D坐标

**输入**：XRHandSubsystem（PICO4 OpenXR接口）

**输出格式**（字符串，通过ZMQ发送）：
```
<type_marker>:x1,y1,z1|x2,y2,z2|...|x26,y26,z26:
```
- `type_marker`：`"relative"`（相对模式）或 `"absolute"`（绝对模式）
- 坐标部分：26个关节的xyz坐标，用`|`分隔，每个坐标用`,`分隔
- 末尾的`:`为序列化终止符

**26关节顺序**（XRHandJointID）：
| 索引 | 关节名称 | 说明 |
|------|----------|------|
| 0 | Wrist | 手腕 |
| 1 | Palm | 手掌 |
| 2 | ThumbMetacarpal | 拇指掌骨 |
| 3 | ThumbProximal | 拇指近端 |
| 4 | ThumbDistal | 拇指远端 |
| 5 | ThumbTip | 拇指尖端 |
| 6 | IndexMetacarpal | 食指掌骨 |
| 7 | IndexProximal | 食指近端 |
| 8 | IndexIntermediate | 食指中间 |
| 9 | IndexDistal | 食指远端 |
| 10 | IndexTip | 食指尖端 |
| 11-15 | Middle系列 | 中指（同食指结构） |
| 16-20 | Ring系列 | 无名指（同食指结构） |
| 21-25 | Little系列 | 小指（同食指结构） |

**手势控制**：
- 左手食指捏合 → 相对数据模式（StreamRelativeData）
- 左手中指捏合 → 绝对数据模式（StreamAbsoluteData）
- 左手无名指捏合 → 停止遥操作

**转发**：`BeaVR-app/BeaVR-Unity/Assets/Scripts/Network/NetMQController.cs`
- 使用NetMQ PushSocket发送
- 右手发送到端口8087，左手发送到端口8110

### 3.2 环节二：BeaVR-bot数据接收

**文件**：`beavr-bot/src/beavr/teleop/components/detector/vr/pico4.py`

**功能**：从PICO4 Unity应用接收原始手部数据，解析后发布

**输入**：ZMQ PULL套接字，接收原始字节数据

**输出**：`InputFrame`对象，通过ZMQ PUB发布

**InputFrame数据结构**：
```python
@dataclass(frozen=True)
class InputFrame:
    timestamp_s: float          # 时间戳（秒）
    hand_side: HandSide         # 手侧 ("left" / "right")
    keypoints: Sequence[Tuple[float, float, float]]  # 26个关节xyz坐标（扁平化78个float）
    is_relative: bool           # 是否为相对模式
    frame_vectors: Optional[Tuple[...]]  # 方向帧向量（此阶段为None）
```

**端口映射**：
| 数据 | 接收端口 | 发布端口 | 发布Topic |
|------|----------|----------|-----------|
| 右手关键点 | 8087 | 8088 | "right" |
| 左手关键点 | 8110 | 8088 | "left" |
| 按钮事件 | 8095 | 8088 | "button" |
| 暂停/恢复 | 8100 | 8088 | "pause" |

### 3.3 环节三：坐标变换

**文件**：`beavr-bot/src/beavr/teleop/components/detector/vr/keypoint_transform.py`

**功能**：对手部关键点进行坐标变换和平滑处理

**变换流程**：
1. **平移**：以手腕为原点，`translated_coords = hand_coords - hand_coords[0]`
2. **旋转**：应用旋转矩阵（目前使用单位矩阵`np.eye(3)`，保持原始坐标不变）
3. **方向帧计算**：基于食指/中指/小指掌指关节计算手部朝向的3D坐标系
4. **滑动平均**：对坐标和方向帧进行时域平滑（窗口大小5帧）
5. **正交化**：Gram-Schmidt正交化确保方向帧向量正交

**输出**：`InputFrame`对象（增加`frame_vectors`字段）

**方向帧结构**：`[wrist_pos, x_vec, y_vec, z_vec]`
- `wrist_pos`：手腕在世界坐标系下的位置（3D向量）
- `x_vec, y_vec, z_vec`：手部朝向的三个正交基向量（各3D向量）

**端口映射**：
| 数据 | 订阅端口 | 发布端口 | 发布Topic |
|------|----------|----------|-----------|
| 右手变换坐标 | 8088 | 8092 | "right_transformed_hand_coords" |
| 右手方向帧 | 8088 | 8092 | "right_transformed_hand_frame" |
| 左手变换坐标 | 8088 | 8093 | "left_transformed_hand_coords" |
| 左手方向帧 | 8088 | 8093 | "left_transformed_hand_frame" |

### 3.4 环节四：运动重定向（Sysmo32Operator）

**文件**：`beavr-bot/src/beavr/teleop/components/operator/robots/sysmo32_operator.py`

**基类**：`XArmOperator`（继承全部遥操作逻辑，仅替换坐标系变换矩阵）

**功能**：将手部运动映射到SYSMO-32机器人末端执行器的笛卡尔空间目标

**核心算法**（继承自XArmOperator）：
1. **获取手部方向帧**：从keypoint_transform.py订阅`transformed_hand_frame`
2. **转换为齐次矩阵**：4x3帧 → 4x4齐次变换矩阵
3. **计算手部相对运动**：`H_HT_HI = inv(H_hand_init) @ H_hand_moving`
4. **坐标系变换**：
   - 旋转部分：`h_ht_hi_r = inv(H_R_V)[:3,:3] @ h_ht_hi[:3,:3] @ H_R_V[:3,:3]`
   - 平移部分：`h_ht_hi_t = inv(H_T_V)[:3,:3] @ h_ht_hi[:3,3] * resolution_scale`
5. **计算目标位姿**：`H_target = H_robot_init @ relative_affine_in_robot_frame`
6. **互补滤波**：平滑目标位姿，减少抖动
7. **输出**：`CartesianTarget`对象

**CartesianTarget数据结构**：
```python
@dataclass(frozen=True)
class CartesianTarget:
    timestamp_s: float
    hand_side: HandSide
    frame_id: str                              # 参考坐标系（"base"）
    position_m: Tuple[float, float, float]     # 目标位置 (x,y,z) 米
    orientation_xyzw: Tuple[float, float, float, float]  # 目标姿态四元数 (x,y,z,w)
```

**SYSMO-32坐标系变换矩阵**：
- `H_R_V_SYSMO32`：SYSMO-32机器人基座坐标系到VR坐标系的变换矩阵
  ```
  [[0, 0, 1, 0],
   [0,-1, 0, 0],
   [-1, 0, 0, 0],
   [0, 0, 0, 1]]
  ```
- `H_T_V_SYSMO32_RIGHT`：右手手部追踪坐标系到VR坐标系的变换矩阵
  ```
  [[0,-1, 0, 0],
   [0, 0,-1, 0],
   [-1, 0, 0, 0],
   [0, 0, 0, 1]]
  ```
- `H_T_V_SYSMO32_LEFT`：左手手部追踪坐标系到VR坐标系的变换矩阵
  ```
  [[0, 1, 0, 0],
   [0, 0, 1, 0],
   [-1, 0, 0, 0],
   [0, 0, 0, 1]]
  ```

**端口映射**：
| 数据 | 订阅端口 | 发布端口 | 发布Topic |
|------|----------|----------|-----------|
| 右手末端命令 | 8092 | 10011 | "endeff_coords" |
| 左手末端命令 | 8093 | 10013 | "endeff_coords" |

### 3.5 环节五：SYSMO-32机器人接口

**文件**：`beavr-bot/src/beavr/teleop/components/interface/robots/sysmo32_robot.py`

**功能**：接收笛卡尔空间目标命令，驱动SYSMO-32双臂机器人运动

**与XArm7Robot的区别**：
| 特性 | XArm7Robot | Sysmo32Robot |
|------|-----------|--------------|
| 每臂关节数 | 7 | 6 |
| 控制器 | DexArmControl (XArm SDK) | MockSysmo32Control (仿真) |
| 双臂结构 | 独立IP | 共享base_link |
| 端口偏移 | 基准 | +2(右臂), +4(左臂) |

**输入**：`CartesianTarget`对象（位置+四元数姿态）

**处理流程**：
1. 接收`CartesianTarget`命令
2. 通过MockSysmo32Control驱动机器人运动（仿真模式）
3. 发布机器人当前状态（关节位置、笛卡尔位姿等）

**状态发布格式**：
```python
{
    "joint_states": {"joint_position": [...6个...], "timestamp": float},
    "operator_cartesian_states": {"cartesian_position": [x,y,z], "timestamp": float},
    "sysmo32_cartesian_states": {"cartesian_position": [x,y,z], "timestamp": float},
    "commanded_cartesian_state": {"commanded_cartesian_position": [x,y,z,qx,qy,qz,qw], "timestamp": float},
    "joint_angles_rad": [...6个...],
    "timestamp": float
}
```

**端口映射**：
| 数据 | 订阅端口 | 发布端口 | 发布Topic |
|------|----------|----------|-----------|
| 右臂末端命令 | 10011 | 10012 | "endeff_coords" / "endeff_homo" |
| 左臂末端命令 | 10013 | 10014 | "endeff_coords" / "endeff_homo" |

## 4. 端口总览

| 端口号 | 常量名 | 用途 | 方向 |
|--------|--------|------|------|
| 8087 | RIGHT_HAND_PICO4_PORT | 右手原始数据 | Unity→Bot |
| 8110 | LEFT_HAND_PICO4_PORT | 左手原始数据 | Unity→Bot |
| 8088 | KEYPOINT_STREAM_PORT | 关键点发布 | pico4→transform |
| 8092 | KEYPOINT_TRANSFORM_PORT | 右手变换后数据 | transform→operator |
| 8093 | LEFT_KEYPOINT_TRANSFORM_PORT | 左手变换后数据 | transform→operator |
| 8095 | RESOLUTION_BUTTON_PORT | 分辨率按钮 | Unity→Bot |
| 8100 | TELEOP_RESET_PORT | 暂停/恢复 | Unity→Bot |
| 10011 | SYSMO32右臂命令 | 右臂末端命令 | operator→robot/mujoco |
| 10012 | SYSMO32右臂状态 | 右臂末端状态 | robot→operator |
| 10013 | SYSMO32左臂命令 | 左臂末端命令 | operator→robot/mujoco |
| 10014 | SYSMO32左臂状态 | 左臂末端状态 | robot→operator |

## 5. 数据类型接口总结

| 数据类型 | 定义文件 | 用途 | 环节 |
|----------|----------|------|------|
| `InputFrame` | detector_types.py | 手部关键点帧 | pico4→transform→operator |
| `CartesianTarget` | operator_types.py | 笛卡尔空间目标 | operator→robot |
| `JointTarget` | operator_types.py | 关节空间目标 | operator→robot(LEAP) |
| `CartesianState` | interface_types.py | 笛卡尔状态反馈 | robot→recorder |
| `ButtonEvent` | detector_types.py | 按钮事件 | pico4→operator |
| `SessionCommand` | detector_types.py | 会话命令(暂停/恢复) | pico4→operator/robot |

## 6. SYSMO-32机器人URDF结构

**URDF路径**：`/home/likunwei/monkey_king/src/sysmo_description/urdf/sysmo32.urdf`

**机器人名称**：SYSMO-32

**结构**：双臂上半身，每臂6自由度，共12个旋转关节

**关节列表**：
| 关节名称 | 类型 | 旋转轴 | 限位(rad) | 说明 |
|----------|------|--------|-----------|------|
| left_shoulder_pitch_joint | revolute | Y | [-3.14, 3.14] | 左肩俯仰 |
| left_shoulder_roll_joint | revolute | X | [-0.099, 2.8] | 左肩横滚 |
| left_shoulder_yaw_joint | revolute | Y | [-3.14, 3.14] | 左肩偏航 |
| left_elbow_joint | revolute | Z | [-1.57, 1.57] | 左肘 |
| left_wrist_yaw_joint | revolute | Y | [-3.14, 3.14] | 左腕偏航 |
| left_wrist_pitch_joint | revolute | Z | [-1.57, 1.57] | 左腕俯仰 |
| right_shoulder_pitch_joint | revolute | Y | [-3.14, 3.14] | 右肩俯仰 |
| right_shoulder_roll_joint | revolute | X | [-2.8, 0.096] | 右肩横滚 |
| right_shoulder_yaw_joint | revolute | Y | [-3.14, 3.14] | 右肩偏航 |
| right_elbow_joint | revolute | Z | [-1.57, 1.57] | 右肘 |
| right_wrist_yaw_joint | revolute | Y | [-3.14, 3.14] | 右腕偏航 |
| right_wrist_pitch_joint | revolute | Z | [-1.57, 1.57] | 右腕俯仰 |

**末端执行器**：
- 左臂末端：`left_arm_J6_Link`
- 右臂末端：`right_arm_J6_Link`

**运动链**：
```
                    base_link
                   /         \
  left_shoulder_pitch_joint   right_shoulder_pitch_joint
            |                          |
     left_arm_J1_Link          right_arm_J1_Link
            |                          |
  left_shoulder_roll_joint    right_shoulder_roll_joint
            |                          |
     left_arm_J2_Link          right_arm_J2_Link
            |                          |
   left_shoulder_yaw_joint    right_shoulder_yaw_joint
            |                          |
     left_arm_J3_Link          right_arm_J3_Link
            |                          |
      left_elbow_joint          right_elbow_joint
            |                          |
     left_arm_J4_Link          right_arm_J4_Link
            |                          |
    left_wrist_yaw_joint      right_wrist_yaw_joint
            |                          |
     left_arm_J5_Link          right_arm_J5_Link
            |                          |
   left_wrist_pitch_joint    right_wrist_pitch_joint
            |                          |
     left_arm_J6_Link          right_arm_J6_Link
       (左臂末端)                  (右臂末端)
```

## 7. 关键代码文件索引

| 文件路径 | 功能 | 数据流环节 |
|----------|------|-----------|
| `BeaVR-app/.../GestureDetectorXR.cs` | PICO4手部数据采集 | 环节一 |
| `BeaVR-app/.../NetMQController.cs` | ZMQ网络转发 | 环节一 |
| `beavr-bot/.../detector/vr/pico4.py` | 数据接收与解析 | 环节二 |
| `beavr-bot/.../detector/vr/keypoint_transform.py` | 坐标变换与平滑 | 环节三 |
| `beavr-bot/.../operator/robots/sysmo32_operator.py` | SYSMO-32运动重定向 | 环节四 |
| `beavr-bot/.../interface/robots/sysmo32_robot.py` | SYSMO-32机器人接口 | 环节五 |
| `beavr-bot/.../simulation/mujoco_sim.py` | MuJoCo仿真环境 | 环节六 |
| `beavr-bot/.../configs/robots/sysmo32_config.py` | SYSMO-32完整配置 | 配置 |
| `beavr-bot/.../configs/robots/sysmo_mujoco_config.py` | MuJoCo仿真配置 | 配置 |
| `beavr-bot/.../detector/detector_types.py` | 数据类型定义 | 通用 |
| `beavr-bot/.../operator/operator_types.py` | Operator数据类型 | 通用 |
| `beavr-bot/.../interface/interface_types.py` | Interface数据类型 | 通用 |
| `beavr-bot/.../configs/constants/robots.py` | 机器人常量 | 配置 |
| `beavr-bot/.../configs/constants/network.py` | 网络常量 | 配置 |
| `beavr-bot/.../configs/constants/ports.py` | 端口常量 | 配置 |

## 8. MuJoCo仿真环境

### 8.1 概述

MuJoCo仿真环境是数据流的仿真终端，替代物理机器人，在MuJoCo物理引擎中渲染SYSMO-32双臂机器人。它订阅Sysmo32Operator发布的CartesianTarget命令，通过逆运动学(IK)求解关节角度，驱动仿真中的机器人双臂运动。

### 8.2 数据流位置

```
sysmo32_operator.py → [MuJoCo仿真: mujoco_sim.py]
                       ZMQ SUB (10011/10013) → IK求解 → MuJoCo仿真渲染
```

### 8.3 核心组件

**文件**：`beavr-bot/src/beavr/teleop/components/simulation/mujoco_sim.py`

**类**：`MuJoCoSysmoSimulator`

**主要功能**：
1. 加载SYSMO-32 URDF模型到MuJoCo
2. 动态添加末端执行器site（left_endeff, right_endeff）
3. 订阅Sysmo32Operator发布的CartesianTarget命令
4. 使用MuJoCo内置IK求解器计算关节角度
5. 驱动仿真中的机器人双臂运动
6. 提供可视化渲染窗口

### 8.4 IK求解算法

使用MuJoCo内置的Jacobian伪逆方法：
1. 获取当前末端执行器位姿
2. 计算位姿误差（位置误差+姿态误差）
3. 计算雅可比矩阵
4. 使用阻尼最小二乘法求解关节角度增量：`delta_q = (J^T J + λ²I)^{-1} J^T error`
5. 更新关节角度
6. 重复直到收敛或达到最大迭代次数

### 8.5 端口配置

| 数据 | 订阅端口 | 发布端口 | Topic |
|------|----------|----------|-------|
| 右手末端命令 | 10011 | - | "endeff_coords" |
| 左手末端命令 | 10013 | - | "endeff_coords" |

### 8.6 启动方式

```bash
# 启动SYSMO-32遥操作 + MuJoCo仿真
bash scripts/run_sysmo32.sh --mujoco

# 仅启动MuJoCo仿真（带渲染窗口）
bash scripts/run_mujoco_sim.sh

# 仅启动MuJoCo仿真（无渲染模式）
bash scripts/run_mujoco_sim.sh --no-render

# 使用teleop.py直接启动
python teleop.py --robot_name=sysmo32 --laterality=bimanual
```

### 8.7 配置文件

**文件**：`beavr-bot/src/beavr/teleop/configs/robots/sysmo_mujoco_config.py`

```python
@dataclass
class MuJoCoSimConfig:
    host: str = "192.168.1.133"
    urdf_path: str = "configs/robots/sysmo32.urdf"
    right_endeff_subscribe_port: int = 10011  # XARM_ENDEFF_SUBSCRIBE_PORT + 2
    left_endeff_subscribe_port: int = 10013   # XARM_ENDEFF_SUBSCRIBE_PORT + 4
    render: bool = True
    simulation_mode: bool = True
```

### 8.8 URDF模型处理

MuJoCo加载URDF时会自动转换为内部XML格式。为了添加末端执行器site，仿真器执行以下步骤：
1. 加载URDF文件到临时MuJoCo模型
2. 使用`mj_saveLastXML`导出为MuJoCo XML格式
3. 在XML中为`left_arm_J6_Link`和`right_arm_J6_Link`添加site定义
4. 重新加载修改后的XML字符串

添加的site定义：
```xml
<body name="left_arm_J6_Link">
    <site name="left_endeff" pos="0 0.07 0" rgba="0 1 0 1" size="0.02"/>
</body>
<body name="right_arm_J6_Link">
    <site name="right_endeff" pos="0 -0.07 0" rgba="1 0 0 1" size="0.02"/>
</body>
```

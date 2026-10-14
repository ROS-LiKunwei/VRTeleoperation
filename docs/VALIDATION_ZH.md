# FA C++ v1.1：MuJoCo 仿真、模式切换与验证

## 结论与运行条件

统一入口支持 `control_backend:=mujoco` 与 `control_backend:=real`。MuJoCo 使用原生 C API，控制、IK、MinSnap、模型与真机模式共用；图形窗口独立运行。原 `/home/likunwei/dataCollection` 的 3079 个受保护文件复核通过，变化文件数为零。

推荐继续使用 `precision_200hz`：200 Hz 控制、1000 Hz 采样、20 ms 跟踪轨迹，IK 接受门限 0.2 mm / 0.002 rad，速度上限 0.5 rad/s、加速度上限 3 rad/s²。跟踪规划预留 4 ms；初始化长轨迹独立预留 50 ms，避免初始化的完整约束检查超过跟踪预算。

本次 20 s 低速 MuJoCo 模型测试中，左右臂最大位置误差为 **0.2408 / 0.2863 mm**，最大姿态误差为 **0.0006426 / 0.0006937 rad**，满足该条件下的 **1 mm / 0.01 rad** 门限。

公开 launch 的 30 s 跟踪段测得控制循环 **200.000 Hz**，观察到 MuJoCo 反馈 **998.956 Hz**。测试通过不等于硬实时截止时间或物理 TCP 精度承诺；本次没有启动真机或重跑 v1.0 的 10 分钟基准。

## 架构与切换命令

```text
PICO → 200 Hz C++ 映射/IK → 带版本 JointTarget
     → 独立 MinSnap 规划 → 1000 Hz 采样
         ├─ real → 已有 FA 驱动 → joint_states
         └─ mujoco → qpos / mj_forward → joint_states
                                            └→ 独立 30 FPS GLFW 窗口
```

```bash
source /opt/ros/humble/setup.bash
source /home/likunwei/VRTeleoperation-cpp/cpp_ws/install/setup.bash

# MuJoCo，默认 /fa_mujoco 命名空间和可视化窗口
ros2 launch fa_teleop teleop.launch.py control_backend:=mujoco enable_motion:=true

# 真机，默认根命名空间，使用已有硬件驱动
ros2 launch fa_teleop teleop.launch.py control_backend:=real enable_motion:=true
```

切换模式需停止当前 launch 后重新启动。无窗口加 `render:=false`。两种模式均需 PICO 显式 High 使能、初始化和轴标定；默认 `enable_motion:=false`。仿真禁止根命名空间。真机模式仅启动控制与执行器，不会启动硬件驱动。

MuJoCo 执行最终 16 维关节样本，不重复 IK 或插值。与原 Python 相同，这是固定基座运动学镜像：设置 `qpos` 后 `mj_forward`，没有接触动力学；手指仅显示开合命令。C API 渲染流程参考 [MuJoCo 官方可视化文档](https://mujoco.readthedocs.io/en/stable/programming/visualization.html)。

## 严格对应的版本

完整 release ID：`943f80ce5bbe51390bb98797a75a398e31f55d649d2dd03845c8cb4e2bc8d107`。

| 组件 | 完整版本 | 来源提交 |
|---|---|---|
| config | `1.1.0+e08b171bde82` | `new implementation / SDK` |
| fa_model | `1.0.0+8647fc6b4d1b` | `7f03b73ad3b88be3c8e6096f320b67d7ff39517f` |
| ik | `1.0.0+04b32d71a2de` | `b1560aae2807f992b1e8fa6f1759527ee600e968` |
| min_snap | `1.0.0+70c763e3fc28` | `b1560aae2807f992b1e8fa6f1759527ee600e968` |
| mujoco_runtime | `3.8.0+310954fe9571` | `new implementation / SDK` |
| teleop | `1.1.0+a9e4596ff88a` | `new implementation / SDK` |

bundle 逐文件锁定源码、启动文件、配置、URDF/SRDF 和网格，编译嵌入 release ID。IK、MinSnap 静态链接；MuJoCo 校验头文件版本、实际加载共享库版本与 SHA，C 运行库随安装包复制。节点身份为 `release SHA:profile SHA:real或mujoco`，阻止组件混用。

构建产物与动态依赖见 [BUILD_RECEIPT.json](../validation/BUILD_RECEIPT.json)，完整组件清单见 [COMPONENT_VERSIONS.json](../provenance/COMPONENT_VERSIONS.json)。

## 模型内精度测试

使用锁定 FA URDF，保留 pelvis 与左右 TCP。200 组关节姿态对照 MuJoCo 与 Pinocchio FK，位置最大差 **4.996e-16 m**，姿态最大差 **7.360e-16 rad**。16 关节按名称映射，包含颈部；越界与 NaN 输入拒绝并保持上一状态。

数值测试共 4000 个 5 ms 周期，覆盖 20 s 模拟时间；14 个臂关节构造幅度 0.02 rad、频率 0.1 Hz 的可达正弦目标，腰部固定。每周期执行双臂 IK、MinSnap 规划及五次 1 ms MuJoCo 更新，比较该周期目标与周期末实际 TCP；没有通过时间平移消除延迟。

| 项目 | 结果 |
|---|---|
| 左 / 右最大位置误差 | 0.2408 / 0.2863 mm |
| 左 / 右最大姿态误差 | 0.0006426 / 0.0006937 rad |
| IK / 规划失败步数 | 0 |
| 单周期最大工作耗时，含 5 次 MuJoCo 更新 | 2.3967 ms |

结果只覆盖上述低速、可达、固定腰部条件。0.5 rad/s 是限制值，不是已验证该速度下达到 1 mm 的证据；没有验证负载、碰撞、接触、力反馈或真实 TCP 标定误差。

## 公开启动入口与故障测试

测试使用 localhost、domain 196、隔离命名空间及独立 ZMQ 端口；PICO 输入模拟为 60 Hz。先检查未 High 使能时无命令且模型位于零位，再执行初始化、跟踪、Low 暂停、High 恢复及输入断流暂停。

| 项目 | 结果 |
|---|---|
| 持续跟踪时间 | 30.000 s |
| 控制循环频率 | 200.000 Hz |
| 订阅端观察到的反馈频率 | 998.956 Hz |
| 后端计数估算反馈频率 | 999.790 Hz |
| 全流程关节命令数 | 42280 |
| 最大相邻关节输出步长 | 0.00103683 rad |
| 控制线程最大计算 / 唤醒延迟 | 0.5393 / 0.9406 ms |
| 控制计算超时 | 0 次 |
| 根话题反馈 / 命令数量 | 0 / 0 |

控制频率使用诊断计数和对应消息接收时间的相同边界计算；诊断约 1 Hz，结果是区间频率，不代表每周期都准时。端到端测试 `simulation_test` 省略需要真人移动的轴标定步骤；数值精度使用 `precision_200hz`，Python/C++ 映射另有对照测试。

- CTest：6 组通过、0 组失败；核心合约 141 项。包含数值回归、Python 坐标映射对照、版本篡改/迁移、MuJoCo 模型和 launch 模式选择。
- real 接口的隔离执行器测试：反馈丢失锁存、反馈恢复不自动续跑、配置混用拒绝、重复命令发布者拒绝。
- mujoco 接口测试：上述保护及后端心跳丢失锁存、恢复不自动续跑、后端身份不匹配拒绝。
- 动态库替换：只篡改临时 MuJoCo 库副本，实际加载 SHA 不符时启动拒绝，原库未修改。
- GLFW/OpenGL：隐藏窗口渲染 10 帧正常退出；[渲染预览](../validation/mujoco_preview.png)。

## 原始证据与复现

- [MuJoCo 模型与精度](../validation/mujoco_model.json)
- [公开 launch 闭环](../validation/mujoco_integration/result.json)
- [real 接口保护](../validation/executor_guards_v1_1/result.json)
- [mujoco 接口与后端保护](../validation/mujoco_guards/result.json)
- [运行库锁定](../validation/mujoco_runtime_lock.json)
- [窗口烟雾测试](../validation/mujoco_viewer.json)
- [CTest](../validation/CTEST.json)
- [原目录保护复核](../validation/SOURCE_INTEGRITY.json)

v1.0 配置选择与 10 分钟基准见 [历史报告](VALIDATION_V1_0_ZH.md) 和 `validation/v1_0/`，属于旧 release，不能替代当前版本测试。本次沿用该配置并补充实际 MuJoCo 验证，不宣称全工作空间或所有负载下的全局最优。部署、复位与复现命令见 [DEPLOYMENT_ZH.md](DEPLOYMENT_ZH.md)。

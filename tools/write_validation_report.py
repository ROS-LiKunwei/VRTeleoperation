#!/usr/bin/env python3
"""Produce the reviewable report from final-machine-readable test results."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
load = lambda name: json.loads((root / name).read_text())
versions = load("provenance/COMPONENT_VERSIONS.json")
realtime = load("validation/realtime_600s.json")
sweep = load("validation/duration_sweep/selection.json")
integration = load("validation/integration_final/result.json")
guards = load("validation/executor_guards/result.json")
ctest = load("validation/CTEST.json")
source = load("validation/SOURCE_INTEGRITY.json")
assert realtime["release_id"] == versions["release_id"] == guards["release_id"] == integration["teleop"]["release_id"]
rows = "\n".join(f"| {name} | `{c['full_version']}` | `{c['origin_commit'] or 'new implementation'}` |" for name, c in versions["components"].items())
experiments = "\n".join(f"| {r['trajectory_duration_s']*1000:.0f} | {r['position_max_m'][0]*1000:.4f} | {r['position_max_m'][1]*1000:.4f} | {max(r['orientation_max_rad']):.6f} | {r['compute_ms_p99']:.4f} | {r['failed_steps']} |" for r in sweep["candidates"])
strict = "通过" if realtime["strict_all_deadlines_pass"] else "未通过；不能声称全部周期满足硬截止时间"
report = f"""# FA C++ 重构测试与推荐配置

## 结论及适用范围

推荐 `precision_200hz` 配置：200 Hz 控制、1000 Hz 轨迹采样、20 ms 期望轨迹时长、IK 位置/姿态接受门限 0.2 mm / 0.002 rad。

在下述简化运动学模型和低速可达轨迹条件下，最终构建的左右臂最大位置误差分别为 **{realtime['position_max_m'][0]*1000:.4f} mm / {realtime['position_max_m'][1]*1000:.4f} mm**，最大姿态误差分别为 **{realtime['orientation_max_rad'][0]:.6f} rad / {realtime['orientation_max_rad'][1]:.6f} rad**，满足模型内 1 mm / 0.01 rad 门限。

10 分钟实时基准的全部截止时间判定：**{strict}**。没有进行物理机器人、外部 TCP 测量、PICO 新传感器帧率或 Android APK 的验证。这里的“推荐”指已测试候选中满足模型内精度条件且最大位置误差最小的配置，不代表所有工作空间、速度和负载下的全局最优。

## 最终版本

完整 release ID：`{versions['release_id']}`。

| 组件 | 完整版本 | 来源提交 |
|---|---|---|
{rows}

FA 模型组同时包含 URDF、SRDF 和必要网格；文件级 SHA 在安装包 `bundle/release.json` 中。SRDF 来自本机 `humanoid_ws/src/fa_moveit2_config`，其所在仓库来源提交为 `b1560aae2807f992b1e8fa6f1759527ee600e968`。IK 源码增加迭代边界的协作式截止时间；MinSnap 原数学内核保留，执行线程和版本接口在新代码中实现。

构建产物与动态依赖的实际 SHA 见 [BUILD_RECEIPT.json](../validation/BUILD_RECEIPT.json)。组件版本清单见 [COMPONENT_VERSIONS.json](../provenance/COMPONENT_VERSIONS.json)。

## 自动化验证

- CTest：{ctest['groups_passed']} 组通过，{ctest['groups_failed']} 组失败；核心合约包含 {ctest['core_contract_checks']} 项检查。
- 原 Python 双手坐标系及 base-frame 位姿映射与 C++ 对照：`1e-12` 容差通过。
- 安装 bundle 移动到另一目录后仍可验证；独立篡改 IK、MinSnap、模型、配置、release 清单或文件清单均拒绝。
- ROS 端到端：显式初始化、跟踪、暂停、恢复、VR 断流暂停与算法版本故障锁存通过。
- 执行器专项：反馈失效锁存、反馈恢复不自动续跑、不同配置拒绝、重复上身命令发布者拒绝均通过。
- 原目录完整性：{source['verified_files']} 个受保护文件复核，变化文件数 {len(source['changed_files'])}。

本次隔离 ROS 端到端测试共观察到 {integration['commands']} 条上身命令；最大相邻关节输出变化为 {integration['max_step_rad']:.8f} rad。控制线程报告最大计算耗时 {integration['teleop']['max_work_ms']:.4f} ms，最大唤醒延迟 {integration['teleop']['max_wakeup_ms']:.4f} ms，计算超时 {integration['teleop']['compute_overruns']} 次。

## 单因素配置比较

固定 IK 门限、迭代预算、速度/加速度约束和模型，仅调整轨迹期望时长；每组运行 4000 步，覆盖 20 s 模拟时间。

| 轨迹时长 ms | 左臂最大位置 mm | 右臂最大位置 mm | 最大姿态 rad | 计算 P99 ms | 失败步数 |
|---|---|---|---|---|---|
{experiments}

四组都满足该轨迹下的模型内精度门限；20 ms 的同时刻位置误差最低，因此作为推荐配置。不同实验运行时桌面负载不同，微小耗时差异不作为算法优劣结论。

## 10 分钟基准

| 项目 | 测量结果 |
|---|---|
| 周期数 | {realtime['steps']} |
| 总耗时 | {realtime['elapsed_s']:.3f} s，含短暂预热 |
| 计算 P99 / 最大值 | {realtime['compute_ms_p99']:.4f} / {realtime['compute_ms_max']:.4f} ms |
| IK P99 | {realtime['ik_ms_p99']:.4f} ms |
| 规划 P99 | {realtime['planning_ms_p99']:.4f} ms |
| 唤醒延迟 P99 / 最大值 | {realtime['wakeup_ms_p99']:.4f} / {realtime['wakeup_ms_max']:.4f} ms |
| 计算超过 5 ms | {realtime['compute_deadline_misses']} 次 |
| 唤醒延迟达到 5 ms | {realtime['wakeup_deadline_misses']} 次 |
| IK/规划失败 | {realtime['failed_steps']} 步 |
| 输出参考峰值关节速度 | {realtime['velocity_max_rad_s']:.6f} rad/s |
| 输出参考峰值关节加速度 | {realtime['acceleration_max_rad_s2']:.6f} rad/s² |
| 双臂位置 RMS | {realtime['position_rms_m']*1000:.4f} mm |

基准使用固定初始姿态附近的已知可达双臂关节正弦路径，单关节幅度 0.02 rad、频率 0.1 Hz，全部 14 个臂关节参与。每个 5 ms 周期包含双臂 IK、MinSnap 重规划和 5 个 1 ms 仿真采样；简化执行模型每 1 ms 更新 `q_actual += 0.25 × (q_reference - q_actual)`。这是模型假设，不是对真实硬件带宽的测量。位置/姿态误差使用同一时刻原目标与实际模型状态比较，没有通过时间平移消除延迟。

配置中的 0.5 rad/s 是安全上限，并不表示已经在该速度下验证 1 mm 精度。200 Hz C++ 周期也不等于 PICO 有 200 Hz 新观测。普通桌面 Linux 的抖动表现见 [ENVIRONMENT.json](../validation/ENVIRONMENT.json)；目标机器人设备必须重新运行基准。

## 原始证据与复现

- [10 分钟完整数据](../validation/realtime_600s.json)
- [全部候选与选择依据](../validation/duration_sweep/selection.json)
- [ROS 端到端结果](../validation/integration_final/result.json)
- [执行器故障专项结果](../validation/executor_guards/result.json)
- [CTest 摘要](../validation/CTEST.json)
- [原目录完整性结果](../validation/SOURCE_INTEGRITY.json)

构建、隔离测试、实际运行和故障复位命令见 [DEPLOYMENT_ZH.md](DEPLOYMENT_ZH.md)。本任务没有启动、修改或替换真实机器人上的运行服务。
"""
(root / "docs/VALIDATION_ZH.md").write_text(report)
print("Wrote docs/VALIDATION_ZH.md")

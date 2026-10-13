# FA C++ 遥操作

本分支将 FA 机器人侧遥操作主链路迁移到 C++17，保留 BeaVR Unity 前端、Python 工具及机器人模型。目标控制周期为 **5 ms / 200 Hz**，MinSnap 轨迹采样为 **1 ms / 1000 Hz**。

推荐配置为 [`precision_200hz.yaml`](cpp_ws/src/fa_teleop/config/precision_200hz.yaml)。最终测试结果和适用条件见 [`docs/VALIDATION_ZH.md`](docs/VALIDATION_ZH.md)。模型内测试与真实机器人精度分别验收。

## 架构

```text
Unity/PICO → C++ ZMQ 接收 → 手腕坐标系 / FA 轴标定 / 相对位姿映射
    → 200 Hz 双臂 IK / FK 复核 / 安全检查
    → 带版本号、配置哈希、时间戳及序号的 JointTarget
    → 独立 MinSnap 规划 → 1000 Hz 轨迹采样 → FA 控制器
```

- `fa_core`：协议、坐标变换、标定、IK 包装、版本检查和规划库。
- `fa_teleop_node`：输入与 ROS 反馈独立接收；单调时钟驱动控制循环。
- `fa_min_snap_node`：规划与采样独立线程；验证后在预定时刻切换连续轨迹。
- `fa_mock_robot`：隔离命名空间中的简化运动学测试模型，不连接物理机器人。
- `beavr_state_bridge.py`：可选的采集状态兼容进程，输出原有 `fa_left/fa_right` 字典，不参与控制。

上身控制命令是 **7 左臂 + 7 右臂 + 2 颈部**；采集动作是 **7 左臂 + 7 右臂 + 2 手部状态**。代码分别处理这两种 16 维数据。

完整设计见 [`FA_CPP_ARCHITECTURE_ZH.md`](FA_CPP_ARCHITECTURE_ZH.md)，实际接口差异见 [`docs/DEPLOYMENT_ZH.md`](docs/DEPLOYMENT_ZH.md)。

## 版本对应

[`provenance/COMPONENT_VERSIONS.json`](provenance/COMPONENT_VERSIONS.json) 记录 IK、MinSnap、FA 模型、控制代码和配置的版本。完整组件版本采用 `语义版本+内容哈希前12位`，清单同时保留完整 SHA-256 和来源提交号。

每次构建都生成安装包中的 `bundle/release.json` 和编译进程序的 release ID。启动会逐文件检查源代码副本、模型和配置；任一文件内容不匹配即拒绝启动。节点间还会比较 `release ID:所选配置SHA-256`，阻止不同算法或配置交换运动命令。IK 和 MinSnap 内核静态链接，不会从旧工作区加载同名算法动态库。

构建产物及实际解析到的系统动态库哈希另外记录在 [`validation/BUILD_RECEIPT.json`](validation/BUILD_RECEIPT.json)。不同 CPU 架构需要在目标设备重新编译和验证。

## 构建与离线测试

环境：ROS 2 Humble、C++17、Eigen、Pinocchio、cppzmq/libzmq、yaml-cpp、OpenSSL、Python 3.10 与 colcon。Python 用于构建、测试及可选采集桥；控制节点本身为 C++。

```bash
cd /home/likunwei/VRTeleoperation-cpp
bash tools/build_cpp.sh
```

脚本执行编译、CTest、部署产物哈希记录和版本清单导出。源码复制来源见 [`cpp_ws/src/fa_teleop/ORIGINS.json`](cpp_ws/src/fa_teleop/ORIGINS.json)。

## 运行

先阅读 [`docs/DEPLOYMENT_ZH.md`](docs/DEPLOYMENT_ZH.md)。默认启动不发送运动命令：

```bash
source /opt/ros/humble/setup.bash
source /home/likunwei/VRTeleoperation-cpp/cpp_ws/install/setup.bash
ros2 launch fa_teleop teleop.launch.py enable_motion:=false
```

开启运动需要 `enable_motion:=true`，随后仍需 PICO 显式使能、初始化与轴标定。此分支的实现和测试没有启动真实机器人。

## BeaVR 副本

`beavr-bot/`、`BeaVR-app/`、`televuer/` 保留本地源码快照。Unity 的新增 `CppPrecisionStream` 默认发送静止时的有效帧，避免 1 mm 去重阈值吞掉细小运动；这不会增加 PICO 传感器采样率。

Unity 仍依赖 PICO Unity Integration SDK **3.4.0**。其原有 Package Manager 本地 SDK 路径需在 Unity 构建机上设置；本次未构建 Android APK。Unity 与 Python 的日志、虚拟环境、数据集和构建缓存不进入本分支。

原目录保护清单：[`provenance/source_snapshot.json`](provenance/source_snapshot.json)。本机可执行只读复核：

```bash
python3 cpp_ws/src/fa_teleop/tools/verify_source.py provenance/source_snapshot.json
```

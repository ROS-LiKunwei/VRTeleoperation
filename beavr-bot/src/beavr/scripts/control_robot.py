# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --robot.cameras='{}' \
    --control.type=teleoperate

# Add the cameras from the robot definition to visualize them:
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate \
    --control.fps=30
```

- Record one episode in order to test replay:
```bash
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/koch_test \
    --control.num_episodes=1 \
    --control.push_to_hub=True
```

- Visualize dataset:
```bash
python beavr/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python beavr/scripts/control_robot.py replay \
    --robot.type=so100 \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=$USER/koch_test \
    --control.episode=0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python beavr/scripts/control_robot.py record \
    --robot.type=so100 \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=$USER/koch_pick_place_lego \
    --control.num_episodes=50 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

- For remote controlled robots like LeKiwi, run this script on the robot edge device (e.g. RaspBerryPi):
```bash
python beavr/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=remote_robot
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command and adding `--control.resume=true`.

- Train on this dataset with the ACT policy:
```bash
python beavr/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_pick_place_lego \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_pick_place_lego \
  --job_name=act_koch_pick_place_lego \
  --device=cuda \
  --wandb.enable=true
```

- Run the pretrained policy on the robot:
```bash
python beavr/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/eval_act_koch_pick_place_lego \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10 \
    --control.push_to_hub=true \
    --control.policy.path=outputs/train/act_koch_pick_place_lego/checkpoints/080000/pretrained_model
```
"""

import logging
import multiprocessing  # Needed for process types (用于启动独立进程来处理遥操作等，避免阻塞主进程)
import os
import time
from dataclasses import asdict
from pprint import pformat

# from safetensors.torch import load_file, save_file
from beavr.lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from beavr.lerobot.common.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset_v21_to_v30
from beavr.lerobot.common.policies.factory import make_policy # 用于实例化AI策略模型
from beavr.lerobot.common.robot_devices.control_configs import (
    ControlPipelineConfig,
    RecordControlConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from beavr.lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,     # 初始化键盘监听（用于控制录制流程）
    log_control_info,
    record_episode,             # 录制单个片段
    reset_environment,          # 重置环境（录制间隙用）
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,              # 录制前的预热阶段
)
from beavr.lerobot.common.robot_devices.robots.utils import (
    Robot,
    make_robot_from_config,     # 根据配置创建机器人对象
)
from beavr.lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from beavr.lerobot.common.utils.utils import init_logging, log_say
from beavr.lerobot.configs import parser

########################################################################################
# Control modes
########################################################################################

# 用于存储遥操作相关的多进程列表，在运行时会被填充
_teleop_processes: list[multiprocessing.Process] | None = None  # Populated at runtime


class _RecordTerminalLogFilter(logging.Filter):
    """Keep record-mode terminal output focused on dataset collection prompts.
        终端日志过滤器。
        在“录制模式”下，底层可能会产生大量刷屏日志。这个过滤器的作用是屏蔽无用信息，
        只让核心的流程提示（如“开始预热”、“正在录制”、“重置环境”等）显示在终端上，保持界面清爽。
    """

    _PROMPT_MESSAGES = (
        "Warmup record",
        "Recording episode",
        "Reset the environment",
        "Re-record episode",
        "No frames recorded",
        "Stop recording",
        "Exiting",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True # 错误日志永远放行
        message = record.getMessage()
        # 只有 INFO 级别且以特定提示语开头的日志才会被输出
        return record.levelno == logging.INFO and message.startswith(self._PROMPT_MESSAGES)


def _configure_record_terminal_logging() -> None:
    """应用上面定义的日志过滤器。"""
    os.environ["BEAVR_RECORD_TERMINAL_QUIET"] = "1"
    logging.getLogger().setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.addFilter(_RecordTerminalLogFilter())


def _episode_buffer_size(dataset: LeRobotDataset) -> int:
    """获取当前录制片段(episode)缓冲区的大小，用于判断是否录到了有效数据。"""
    episode_buffer = getattr(dataset, "episode_buffer", None)
    if not episode_buffer:
        return 0
    return int(episode_buffer.get("size", 0))


def start_teleop_process(
    *,
    robot_name: str | None = None,
    laterality: str | None = None,
    operate: bool | None = None,
    wait_for_exit: bool = False,
):
    """
    启动遥操作（Teleoperation）的辅助进程栈（例如：VR检测器/摄像头/机器人接口）。[Launch the teleoperation helper stack (detector/cameras/robot interface)]
    因为视频流获取、VR手柄信号读取非常消耗资源，通常需要放在独立的进程中运行。
   
    Parameters
    ----------
    robot_name : str | None
        要加载的机器人组合的标识符(例如“leap,xarm7”或“leap”), 如果为*None*,
        我们将尝试从命令行界面(CLI)读取`--teleop_robot_name`(通过“beavr.configs.parser”)。
        如果该函数也不存在,回退到默认的“leap,xarm7”。

    operate : bool | None
        是否启动监听VR Detector并将命令转发给机器人的*Operator*进程。
        当*policy*通过`robot.send_action()`自行发布命令时，将此设置为``False``至关重要——
        同时运行两者会导致PUB/SUB冲突。
        当为*None*时,该值将自动派生:每当在命令行界面(CLI)上提供策略时（通过`--control.policy.`参数的存在来检测），该值将设置为``False``；否则，默认值为``True``。
    """

    global _teleop_processes
    if _teleop_processes is not None:
        logging.info("Teleoperation already running – skipping launch.")
        return

    # 延迟导入（Lazy import）：只有在需要用到遥操作时才导入这些库，加快脚本启动速度。
    from beavr.teleop.components import TeleOperator
    from beavr.teleop.main import MainConfig  # pylint: disable=import-error

    # ------------------------------------------------------------------
    # Resolve *robot_name*
    # ------------------------------------------------------------------
    if robot_name is None:
        # Try CLI override --teleop_robot_name=...
        robot_name = parser.parse_arg("teleop_robot_name")

    if robot_name is None:
        # Fallback for legacy flag --teleop.robot_name=...
        robot_name = parser.parse_arg("teleop.robot_name")

    if robot_name is None:
        # Ultimate fallback – use new multi-robot syntax
        robot_name = "leap,xarm7"

    if laterality is None:
        laterality = parser.parse_arg("teleop.laterality")

    if laterality is None:
        laterality = parser.parse_arg("teleop_laterality")

    if laterality is None:
        laterality = "right"

    # ------------------------------------------------------------------
    # Resolve *operate*
    # Operate 决定是否启动将命令转发给机器人的“Operator”进程。
    # 如果你在运行 AI 策略 (Policy)，AI 自己会发命令，此时要把 operate 设为 False，避免人类和 AI 抢夺控制权。
    # ------------------------------------------------------------------
    if operate is None:
        # CLI override (explicit)
        operate_cli = parser.parse_arg("teleop_operate")
        if operate_cli is not None:
            operate = operate_cli.lower() not in {"0", "false", "no", "off"}

    if operate is None:
        # 隐式启发式规则：如果命令行提供了 policy，就禁用操作员进程
        has_policy_path = parser.get_path_arg("control.policy") is not None
        has_policy_type = parser.get_type_arg("control.policy") is not None
        operate = not (has_policy_path or has_policy_type)

    logging.info(
        "Starting teleoperation helper with robot_name='%s', laterality='%s', operate=%s",
        robot_name,
        laterality,
        operate,
    )

    # 使用新的结构化配置为所选机器人组合构建配置
    # 使用robot_name创建MainConfig，并在teleop.flags中设置操作标志
    main_config = MainConfig(robot_name=robot_name, laterality=laterality)
    main_config.teleop.flags.operate = operate

    logging.info("Instantiating TeleOperator (Draccus version)…")
    # 构建并启动 TeleOperator 进程
    teleop = TeleOperator(main_config)
    _teleop_processes = teleop.get_processes()

    # Start all sub-processes.
    for p in _teleop_processes:
        p.start()

    # Give PUB sockets a moment to come up before the main thread proceeds.
    time.sleep(1)
    # 如果要求等待退出，则阻塞主线程直到子进程结束（通常在纯遥控模式下使用）
    if wait_for_exit:
        try:
            for p in _teleop_processes:
                p.join()
        except KeyboardInterrupt:
            logging.info("Teleoperation interrupted by user.")
        finally:
            stop_teleop_process()


def stop_teleop_process():
    """Terminate the TeleOperator's child processes if they are active."""
    global _teleop_processes
    if _teleop_processes:
        logging.info("Stopping teleoperation processes…")
        for p in _teleop_processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        _teleop_processes = None


@safe_disconnect
def teleoperate():
    """Use the BeaVR teleoperation system by running teleop.py and waiting for it to complete."""
    start_teleop_process(wait_for_exit=True)


@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    """
        数据录制的核心逻辑。
        用于通过遥操作(人控制)或策略模型(AI控制)来收集机器人的传感器数据和动作，并保存为数据集。
    """
    # When recording, the teleoperation system is assumed to be already running.
    # This function is now only responsible for the recording logic itself.
    try:
        if hasattr(robot, "manage_teleop_state"):
            robot.manage_teleop_state = cfg.manage_teleop_state

        # TODO: Add option to record logs
        # 处理数据集初始化：如果是断点续录(resume)，则加载已有数据集；否则新建。
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.repo_id,
                root=cfg.root,
            )
            if len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.num_image_writer_processes,
                    num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
        else:
            # Create empty dataset or load existing saved episodes
            sanity_check_dataset_name(cfg.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.repo_id,
                cfg.fps,
                root=cfg.root,
                robot=robot,
                use_videos=cfg.video,
                image_writer_processes=cfg.num_image_writer_processes,
                image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )

        # 如果配置了策略（Policy），则实例化 AI 模型。如果是人类录制，这里为 None。
        policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
        # 初始化键盘监听器，以便在终端里通过快捷键（如 ESC 退出，左右箭头重录）控制流程
        listener, events = init_keyboard_listener()

        # 录制前的预热：不保存数据，纯粹为了让硬件设备连接、画面同步，并允许操作员先把机器人移动到初始位置。
        # Execute a few seconds without recording to:
        # 1. teleoperate the robot to move it in starting position if no policy provided,
        # 2. give times to the robot devices to connect and start synchronizing,
        # 3. place the cameras windows on screen
        enable_teleoperation = policy is None
        log_say("Warmup record", cfg.play_sounds)
        warmup_record(
            robot,
            events,
            enable_teleoperation,
            cfg.warmup_time_s,
            cfg.display_data,
            cfg.fps,
        )

        recorded_episodes = 0
        # 主录制循环
        while True:
            if recorded_episodes >= cfg.num_episodes:
                break

            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            # 执行单个片段(episode)的录制
            record_episode(
                robot=robot,
                dataset=dataset,
                events=events,
                episode_time_s=cfg.episode_time_s,
                display_data=cfg.display_data,
                policy=policy,
                fps=cfg.fps,
                single_task=cfg.single_task,
            )
            # 键盘事件：用户按下要求重新录制当前片段
            if events["rerecord_episode"]:
                if not events["stop_recording"]:
                    log_say("Reset the environment", cfg.play_sounds)
                    reset_environment(robot, events, cfg.reset_time_s, cfg.fps) # 给时间重置物理环境
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer() # 清空刚刚录的废弃数据
                continue
            # 安全检查：如果没有记录到任何帧    
            if _episode_buffer_size(dataset) == 0:
                log_say(
                    "No frames recorded; episode not saved. Waiting for valid observation/action frames.",
                    cfg.play_sounds,
                )
                dataset.clear_episode_buffer()
                if events["stop_recording"]:
                    break
                if recorded_episodes < cfg.num_episodes:
                    log_say("Reset the environment", cfg.play_sounds)
                    reset_environment(robot, events, cfg.reset_time_s, cfg.fps)
                continue
            # 保存有效片段
            dataset.save_episode()
            recorded_episodes += 1

            if events["stop_recording"]:
                break

            # Execute a few seconds without recording to give time to manually reset the environment.
            # Save first so early-exited episodes are durable before reset or a later Ctrl+C.
            # 每个片段录制完后，留出时间让操作员把物体和机器人放回初始位置
            if recorded_episodes < cfg.num_episodes:
                log_say("Reset the environment", cfg.play_sounds)
                reset_environment(robot, events, cfg.reset_time_s, cfg.fps)
        # 录制结束，清理键盘监听
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        stop_recording(robot, listener, cfg.display_data)
        # 处理数据集格式转换
        if cfg.dataset_format == "v3.0":
            convert_dataset_v21_to_v30(dataset.root)
        elif cfg.dataset_format != "v2.1":
            raise ValueError("control.dataset_format must be either 'v3.0' or 'v2.1'")
        # 选择性推送到 Hugging Face Hub（云端）
        if cfg.push_to_hub:
            dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

        log_say("Exiting", cfg.play_sounds)
        return dataset
    finally:
        # Cleanup is now handled by the main control_robot function
        pass


@safe_disconnect
def replay(
    robot: Robot,
    cfg: ReplayControlConfig,
):
    """
    回放模式。
    读取已录制的数据集中的某一个片段(episode)，并将其中的动作指令(action)直接发送给机器人，让机器人原样执行一遍。
    """
    # TODO: refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO: Add option to record logs

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, episodes=[cfg.episode])
    actions = dataset.hf_dataset.select_columns("action")

    if not robot.is_connected:
        robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    # 按帧率逐帧发送动作
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"]
        robot.send_action(action) # 将当前帧的动作发送给机器人
        # 延时等待以匹配目标帧率(FPS)
        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=cfg.fps)


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    """
        脚本的主入口函数。
        负责初始化日志、解析配置、按需启动遥操作子进程、连接硬件，并路由到对应的模式（遥控、录制、回放）。
    """
    init_logging()
    if isinstance(cfg.control, RecordControlConfig):
        _configure_record_terminal_logging() # 如果是录制模式，开启日志过滤
    logging.info(pformat(asdict(cfg)))

    # Determine whether the teleoperation helper needs to run. Set
    # --teleop.start=false when an external teleop.py stack is already running.
    start_teleop = (
        isinstance(cfg.control, (TeleoperateControlConfig, RecordControlConfig)) and cfg.teleop.start
    )

    if start_teleop:
        start_teleop_process(
            robot_name=getattr(cfg.teleop, "robot_name", None),
            laterality=getattr(cfg.teleop, "laterality", None),
            operate=getattr(cfg.teleop, "operate", None),
        )
        logging.info("Waiting 5s for teleoperation system to initialize before robot connection...")
        # TODO: Improve this logic shouldn't need sleep
        time.sleep(5) # 强行等待5秒，确保遥操作系统硬件和通讯初始化完毕

    # 根据配置实例化机器人对象（控制硬件的接口） [Pass the controller to the robot adapter]
    robot = make_robot_from_config(cfg.robot)

    try:
        # 连接硬件
        if not robot.is_connected:
            robot.connect()
        # 根据配置文件指定的模式执行对应逻辑
        if isinstance(cfg.control, TeleoperateControlConfig):
            # The teleop process was already started, just wait for it to complete.
            global _teleop_processes
            if _teleop_processes:
                logging.info("Waiting for teleoperation process to complete...")
                for p in _teleop_processes:
                    p.join() # 纯遥控模式下，主进程在这里挂起，直到用户终止遥操作

        elif isinstance(cfg.control, RecordControlConfig):
            record(robot, cfg.control)
        elif isinstance(cfg.control, ReplayControlConfig):
            replay(robot, cfg.control)
    except KeyboardInterrupt:
        # 捕获 Ctrl+C 的中断信号
        logging.info("Main control loop interrupted by user.")
    finally:
        # This block ensures cleanup happens on normal exit or interrupt.
        logging.info("Starting cleanup...")
        if robot.is_connected:
            logging.info("Disconnecting robot...")
            robot.disconnect()
        # Always try to stop the teleop process on exit
        stop_teleop_process()
        logging.info("Cleanup complete. Exiting.")


if __name__ == "__main__":
    control_robot()

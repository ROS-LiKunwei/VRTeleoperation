import numpy as np

from beavr.teleop.components.interface.robots.sysmo32_trajectory import (
    Sysmo32ArmTrajectoryConfig,
    Sysmo32ArmTrajectorySmoother,
)


def test_min_snap_smoother_starts_from_current_and_reaches_goal():
    smoother = Sysmo32ArmTrajectorySmoother(
        Sysmo32ArmTrajectoryConfig(
            segment_time_s=0.2,
            min_duration_s=0.1,
            max_joint_velocity_rad_s=tuple([10.0] * 6),
            max_joint_acceleration_rad_s2=tuple([100.0] * 6),
        )
    )
    current = np.zeros(6)
    goal = np.ones(6) * 0.5

    start_sample = smoother.sample(goal, current, now_s=10.0)
    mid_sample = smoother.sample(goal, current, now_s=10.1)
    end_sample = smoother.sample(goal, current, now_s=10.3)

    np.testing.assert_allclose(start_sample, current)
    assert np.all(mid_sample > current)
    assert np.all(mid_sample < goal)
    np.testing.assert_allclose(end_sample, goal)


def test_min_snap_duration_respects_velocity_limit():
    smoother = Sysmo32ArmTrajectorySmoother(
        Sysmo32ArmTrajectoryConfig(
            segment_time_s=0.01,
            min_duration_s=0.01,
            max_joint_velocity_rad_s=tuple([1.0] * 6),
            max_joint_acceleration_rad_s2=tuple([1000.0] * 6),
        )
    )

    smoother.sample(np.ones(6), np.zeros(6), now_s=1.0)

    assert smoother._duration_s >= 2.18


def test_min_snap_replan_threshold_holds_existing_goal():
    smoother = Sysmo32ArmTrajectorySmoother(
        Sysmo32ArmTrajectoryConfig(
            segment_time_s=0.2,
            min_duration_s=0.1,
            replan_threshold_rad=0.1,
        )
    )
    smoother.sample(np.ones(6), np.zeros(6), now_s=1.0)
    first_goal = smoother._goal.copy()

    smoother.sample(np.ones(6) + 0.01, np.zeros(6), now_s=1.05)

    np.testing.assert_allclose(smoother._goal, first_goal)


def test_min_snap_replan_continues_from_active_sample_not_current_joint_state():
    smoother = Sysmo32ArmTrajectorySmoother(
        Sysmo32ArmTrajectoryConfig(
            segment_time_s=0.2,
            min_duration_s=0.1,
            replan_threshold_rad=0.001,
            max_joint_velocity_rad_s=tuple([10.0] * 6),
            max_joint_acceleration_rad_s2=tuple([100.0] * 6),
        )
    )
    current = np.zeros(6)
    first_goal = np.ones(6) * 0.5
    next_goal = np.ones(6) * 0.6

    smoother.sample(first_goal, current, now_s=1.0)
    active_sample = smoother.sample(first_goal, current, now_s=1.1)
    replan_sample = smoother.sample(next_goal, current, now_s=1.1)

    np.testing.assert_allclose(replan_sample, active_sample)
    assert np.all(replan_sample > current)


def test_min_snap_streaming_goal_updates_keep_progressing():
    smoother = Sysmo32ArmTrajectorySmoother(
        Sysmo32ArmTrajectoryConfig(
            segment_time_s=0.2,
            min_duration_s=0.1,
            replan_threshold_rad=0.001,
            max_joint_velocity_rad_s=tuple([10.0] * 6),
            max_joint_acceleration_rad_s2=tuple([100.0] * 6),
        )
    )
    current = np.zeros(6)
    samples = []

    for step in range(12):
        now_s = 1.0 + step * 0.02
        target = np.ones(6) * (0.2 + step * 0.02)
        samples.append(smoother.sample(target, current, now_s=now_s))

    first_motion = np.max(samples[2])
    last_motion = np.max(samples[-1])
    assert first_motion > 0.0
    assert last_motion > first_motion


def test_min_snap_reset_uses_current_joint_state():
    smoother = Sysmo32ArmTrajectorySmoother()
    current = np.arange(6, dtype=np.float64) * 0.1

    smoother.reset(current)
    sample = smoother.sample(current, current, now_s=2.0)

    np.testing.assert_allclose(sample, current)

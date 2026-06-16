import numpy as np

from beavr.teleop.components.interface.robots.sysmo32_trajectory import (
    Sysmo32ArmTrajectoryConfig,
    Sysmo32ArmTrajectorySmoother,
    Sysmo32JerkLimitedServoConfig,
    Sysmo32JerkLimitedServoSmoother,
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


def test_jerk_limited_servo_starts_from_current_and_tracks_target():
    smoother = Sysmo32JerkLimitedServoSmoother(
        Sysmo32JerkLimitedServoConfig(
            max_joint_velocity_rad_s=tuple([2.0] * 6),
            max_joint_acceleration_rad_s2=tuple([4.0] * 6),
            max_joint_jerk_rad_s3=tuple([20.0] * 6),
            omega=20.0,
            damping_ratio=1.0,
        )
    )
    current = np.zeros(6)
    target = np.ones(6) * 0.5

    first = smoother.sample(target, current, now_s=1.0)
    later = smoother.sample(target, current, now_s=1.1)

    np.testing.assert_allclose(first, current)
    assert np.all(later > first)
    assert np.all(later < target)


def test_jerk_limited_servo_respects_velocity_acceleration_and_jerk_limits():
    max_velocity = 0.5
    max_acceleration = 1.0
    max_jerk = 5.0
    dt = 0.02
    smoother = Sysmo32JerkLimitedServoSmoother(
        Sysmo32JerkLimitedServoConfig(
            max_joint_velocity_rad_s=tuple([max_velocity] * 6),
            max_joint_acceleration_rad_s2=tuple([max_acceleration] * 6),
            max_joint_jerk_rad_s3=tuple([max_jerk] * 6),
            omega=50.0,
            damping_ratio=1.0,
        )
    )
    target = np.ones(6) * 2.0
    current = np.zeros(6)
    velocities = []
    accelerations = []

    smoother.sample(target, current, now_s=1.0)
    for step in range(1, 40):
        smoother.sample(target, current, now_s=1.0 + step * dt)
        velocities.append(smoother.velocity)
        accelerations.append(smoother.acceleration)

    velocities = np.asarray(velocities)
    accelerations = np.asarray(accelerations)
    jerks = np.diff(accelerations, axis=0) / dt

    assert np.max(np.abs(velocities)) <= max_velocity + 1e-9
    assert np.max(np.abs(accelerations)) <= max_acceleration + 1e-9
    assert np.max(np.abs(jerks)) <= max_jerk + 1e-9


def test_jerk_limited_servo_disabled_passthrough():
    smoother = Sysmo32JerkLimitedServoSmoother(Sysmo32JerkLimitedServoConfig(enabled=False))
    target = np.ones(6) * 0.4

    sample = smoother.sample(target, np.zeros(6), now_s=1.0)

    np.testing.assert_allclose(sample, target)

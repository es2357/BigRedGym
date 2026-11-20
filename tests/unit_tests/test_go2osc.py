"""Oscillator equations, local force feedback, and masked Go2 resets."""

import math

import pytest
import torch

from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg
from gym.envs.go2.go2osc import Go2Osc
from gym.utils.task_registry import select_backend, task_registry


class OscillatorTestCfg(Go2Cfg):
    class env(Go2Cfg.env):
        num_envs = 4

    class init_state(Go2Cfg.init_state):
        reset_mode = "reset_to_basic"
        pos = [0.0, 0.0, 1.5]

    class control(Go2Cfg.control):
        desired_sim_frequency = 500

    class sim:
        gravity = [0.0, 0.0, -9.81]

    class commands(Go2Cfg.commands):
        var = 0.0

        class ranges(Go2Cfg.commands.ranges):
            lin_vel_x = [-1.0, 0.0, 1.0, 3.0]

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            contact_friction_range = None
            link_mass_scale_range = [1.0, 1.0]

        class episode(Go2Cfg.domain_randomization.episode):
            scale_ranges = {}

    class osc:
        omega = 3.0
        coupling = 1.0
        offset = 1.0
        stop_threshold = 0.5
        omega_stop = 1
        omega_step = 2.0
        omega_slope = 1.0
        omega_max = 4.0
        omega_var = 0.0
        coupling_stop = 4.0
        coupling_step = 1.0
        coupling_slope = 0.0
        coupling_max = 1.0
        coupling_var = 0.0
        process_noise_std = 0.0


@pytest.fixture
def env(request):
    cfg = OscillatorTestCfg()
    cfg.seed = 17
    if hasattr(request, "param"):
        cfg.commands.ranges.lin_vel_x = request.param
    runner_cfg = Go2RunnerCfg()
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = select_backend(cfg, "cpu", "mujoco")
    try:
        yield Go2Osc(cfg, "cpu", True, backend)
    finally:
        backend.close()


def _weight(env):
    gravity = math.sqrt(sum(value**2 for value in env.cfg.sim.gravity))
    return env._backend.link_mass.sum(dim=1, keepdim=True) * gravity


def _paper_foot_indices(env):
    return [
        env.robot_layout.body_names.index(name)
        for name in ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
    ]


def _set_foot_load(env, fraction):
    env.contact_forces.zero_()
    env.contact_forces[:, _paper_foot_indices(env), 2] = _weight(env) * fraction
    env.grf.copy_(env._compute_grf())


def test_buffers_expose_finite_cosine_then_sine_observations(env):
    for name, size in (
        ("oscillators", 4),
        ("oscillators_vel", 4),
        ("oscillator_obs", 8),
        ("grf", 4),
        ("osc_omega", 1),
        ("osc_coupling", 1),
        ("osc_offset", 1),
    ):
        value = getattr(env, name)
        assert value.shape == (env.num_envs, size)
        assert torch.isfinite(value).all()
    expected = torch.cat((env.oscillators.cos(), env.oscillators.sin()), dim=1)
    torch.testing.assert_close(env.oscillator_obs, expected)
    assert torch.all((env.oscillators >= 0) & (env.oscillators < 2 * math.pi))
    assert torch.unique(env.oscillators[0]).numel() == 4


def test_command_parameters_include_stop_threshold_and_reverse_motion(env):
    env.commands[:, 0] = torch.tensor([-0.5, 0.5, 1.0, -3.0])
    env._update_oscillator_parameters()
    torch.testing.assert_close(env.osc_omega[:, 0], torch.tensor([1.0, 1.0, 2.5, 4.0]))
    torch.testing.assert_close(
        env.osc_coupling[:, 0], torch.tensor([4.0, 4.0, 1.0, 1.0])
    )
    torch.testing.assert_close(env.osc_offset[:, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_integer_nominal_frequency_preserves_fractional_command_parameters(env):
    assert isinstance(env.cfg.osc.omega_stop, int)
    assert env.osc_omega.is_floating_point()
    env.commands[:, 0] = 1.25
    env._update_oscillator_parameters()
    torch.testing.assert_close(env.osc_omega, torch.full((env.num_envs, 1), 2.75))


def test_parameter_noise_uses_reference_scale_and_nonnegative_clamps(env, monkeypatch):
    env.cfg.osc.omega_var = 0.25
    env.cfg.osc.coupling_var = 0.25
    env.commands[:, 0] = torch.tensor([0.0, 1.0, 3.0, 0.0])
    draws = iter(
        (
            torch.tensor([[-10.0], [1.0], [4.0], [-1.0]]),
            torch.tensor([[-20.0], [-8.0], [1.0], [4.0]]),
        )
    )
    monkeypatch.setattr(torch, "randn_like", lambda value: next(draws).to(value))

    env._update_oscillator_parameters()

    torch.testing.assert_close(
        env.osc_omega[:, 0], torch.tensor([0.1, 2.75, 5.0, 0.75])
    )
    torch.testing.assert_close(
        env.osc_coupling[:, 0], torch.tensor([0.0, 0.0, 1.25, 5.0])
    )


def test_grf_uses_vector_norm_clipping_and_live_per_environment_mass(env):
    vectors = torch.tensor([[3.0, 4.0, 0.0], [0, 0, 0], [0, 0, -10], [0, 0, 20]])
    env.contact_forces.zero_()
    raw = vectors.unsqueeze(0) * _weight(env).unsqueeze(-1) * 0.1
    env.contact_forces[:, _paper_foot_indices(env)] = raw
    expected = torch.tensor([[0.5, 0.0, 1.0, 1.0]]).expand(env.num_envs, -1)
    torch.testing.assert_close(env._compute_grf(), expected)
    torch.testing.assert_close(env._compute_grf(grf_norm=False), raw.norm(dim=-1))

    scales = torch.full((1, env.num_bodies), 2.0)
    env._backend.set_link_mass_scale(torch.tensor([1]), scales)
    torch.testing.assert_close(env._compute_grf()[1], torch.tensor([0.25, 0, 0.5, 1.0]))
    torch.testing.assert_close(env._compute_grf()[0], expected[0])


def test_paper_leg_order_aligns_named_forces_phase_feedback_and_observations(env):
    assert env.LEG_NAMES == ("FR", "FL", "RR", "RL")
    native_feet = tuple(
        env.robot_layout.body_names[index] for index in env.feet_indices.tolist()
    )
    assert native_feet == ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    assert tuple(env.dof_names[::3]) == (
        "FL_hip_joint",
        "FR_hip_joint",
        "RL_hip_joint",
        "RR_hip_joint",
    )
    env.contact_forces.zero_()
    for name, fraction in (
        ("FR_foot", 0.1),
        ("FL_foot", 0.2),
        ("RR_foot", 0.3),
        ("RL_foot", 0.4),
    ):
        index = env.robot_layout.body_names.index(name)
        env.contact_forces[:, index, 2] = _weight(env)[:, 0] * fraction
    phases = torch.tensor([[0.0, math.pi, math.pi / 2, 3 * math.pi / 2]])
    env.oscillators[:] = phases
    env.osc_omega.fill_(1.0)
    env.osc_coupling.fill_(1.0)
    env.osc_offset.fill_(1.0)

    env._step_oscillators(0.002)

    expected_grf = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).expand(env.num_envs, -1)
    torch.testing.assert_close(env.grf, expected_grf)
    velocity = torch.tensor([[0.8, 1.0, 0.7, 0.6]]) * (2 * math.pi)
    torch.testing.assert_close(env.oscillators_vel, velocity.expand(env.num_envs, -1))
    phases = (phases + velocity * 0.002).remainder(2 * math.pi)
    expected_obs = torch.cat((phases.cos(), phases.sin()), dim=1)
    torch.testing.assert_close(
        env.oscillator_obs, expected_obs.expand(env.num_envs, -1)
    )


@pytest.mark.parametrize("feedback_disabled", ("zero_force", "zero_coupling"))
def test_free_running_oscillators_advance_in_hertz_and_wrap(env, feedback_disabled):
    env.oscillators[:] = torch.tensor([0.0, 0.5, math.pi, 2 * math.pi - 0.01])
    env.osc_omega.fill_(2.5)
    env.osc_coupling.fill_(0.0 if feedback_disabled == "zero_coupling" else 4.0)
    _set_foot_load(env, 0.25 if feedback_disabled == "zero_coupling" else 0.0)
    before = env.oscillators.clone()

    env._step_oscillators(0.01)

    torch.testing.assert_close(
        env.oscillators_vel, torch.full_like(before, 5 * math.pi)
    )
    expected = (before + 5 * math.pi * 0.01).remainder(2 * math.pi)
    torch.testing.assert_close(env.oscillators, expected)


def test_local_force_feedback_changes_each_phase_velocity_and_can_reverse(env):
    env.oscillators[:] = torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    env.osc_omega.fill_(1.0)
    env.osc_coupling.fill_(4.0)
    env.osc_offset.fill_(1.0)
    _set_foot_load(env, 0.25)
    before = env.oscillators.clone()

    env._step_oscillators(0.002)

    expected_velocity = torch.tensor([-2 * math.pi, 0.0, 2 * math.pi, 0.0])
    expected_velocity = expected_velocity.expand_as(before)
    torch.testing.assert_close(
        env.oscillators_vel, expected_velocity, atol=2e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        env.oscillators,
        (before + expected_velocity * 0.002).remainder(2 * math.pi),
        atol=2e-6,
        rtol=1e-5,
    )


def test_process_noise_converts_cycles_per_second_to_radians(env, monkeypatch):
    env.cfg.osc.process_noise_std = 0.25
    env.oscillators.zero_()
    env.osc_omega.fill_(1.0)
    env.osc_coupling.zero_()
    _set_foot_load(env, 0.0)
    draws = torch.tensor([[1.0, -2.0, 0.0, 4.0]]).expand_as(env.oscillators)
    monkeypatch.setattr(torch, "randn_like", lambda value: draws.to(value))

    env._step_oscillators(0.002)

    velocity = torch.tensor([[1.25, 0.5, 1.0, 2.0]]) * (2 * math.pi)
    torch.testing.assert_close(env.oscillators_vel, velocity.expand_as(draws))
    torch.testing.assert_close(env.oscillators, (velocity * 0.002).expand_as(draws))


def test_physics_hook_advances_five_times_per_control_step(env, monkeypatch):
    env.commands[:] = torch.tensor([1.0, 0.0, 0.0])
    before = env.oscillators.clone()
    timesteps = []
    advance = env._step_oscillators

    def record_step(dt=None):
        timesteps.append(env.dt / env.cfg.control.decimation if dt is None else dt)
        advance(dt)

    monkeypatch.setattr(env, "_step_oscillators", record_step)
    env.step()

    assert timesteps == pytest.approx([0.002] * 5)
    expected = (before + 5 * math.pi * 0.01).remainder(2 * math.pi)
    torch.testing.assert_close(env.oscillators, expected)
    assert torch.isfinite(env.root_states).all()


def test_partial_reset_preserves_other_environments_and_clears_feedback(env):
    env.oscillators_vel.fill_(123.0)
    env.grf.fill_(0.7)
    env.dof_pos_target.fill_(0.1)
    env.dof_pos_history.fill_(0.25)
    mask = torch.tensor([True, False, True, False])
    names = (
        "oscillators",
        "oscillators_vel",
        "oscillator_obs",
        "grf",
        "osc_omega",
        "osc_coupling",
        "osc_offset",
        "dof_pos_target",
        "dof_pos_history",
    )
    before = {name: getattr(env, name).clone() for name in names}

    env._reset_idx(mask)

    for name, value in before.items():
        torch.testing.assert_close(getattr(env, name)[~mask], value[~mask])
    assert not torch.equal(env.oscillators[mask], before["oscillators"][mask])
    assert torch.count_nonzero(env.grf[mask]) == 0
    assert torch.count_nonzero(env.dof_pos_target[mask]) == 0
    assert torch.count_nonzero(env.dof_pos_history[mask]) == 0
    torch.testing.assert_close(
        env.oscillators_vel[mask],
        (2 * math.pi * env.osc_omega[mask]).expand(-1, 4),
    )
    expected = torch.cat((env.oscillators.cos(), env.oscillators.sin()), dim=1)
    torch.testing.assert_close(env.oscillator_obs, expected)


def test_empty_reset_and_command_masks_preserve_state_and_random_generator(env):
    before = {
        name: value.clone()
        for name, value in vars(env).items()
        if isinstance(value, torch.Tensor)
    }
    random_state = torch.get_rng_state().clone()
    mask = torch.zeros(env.num_envs, dtype=torch.bool)

    env._reset_idx(mask)
    env._resample_commands(mask)

    for name, value in before.items():
        assert torch.equal(getattr(env, name), value), name
    assert torch.equal(torch.get_rng_state(), random_state)


@pytest.mark.parametrize("env", [[1.25]], indirect=True)
def test_command_resampling_updates_only_selected_environments(env):
    env.commands.fill_(99.0)
    mask = torch.tensor([False, True, False, True])
    previous_omega = env.osc_omega.clone()

    env._resample_commands(mask)

    assert torch.all(env.commands[~mask] == 99.0)
    assert torch.all(env.commands[mask, 0] == 1.25)
    assert torch.all(env.commands[mask, 1].abs() <= 1.0)
    assert torch.all(env.commands[mask, 2].abs() <= 3.0)
    torch.testing.assert_close(env.osc_omega[~mask], previous_omega[~mask])
    torch.testing.assert_close(env.osc_omega[mask], torch.full((2, 1), 2.75))


def test_phase_rewards_penalize_swing_loading_and_reward_stance_loading(env):
    env.oscillators[:2].fill_(math.pi / 2)
    env.oscillators[2:].fill_(3 * math.pi / 2)
    _set_foot_load(env, 0.25)

    swing = env._reward_swing_grf()
    stance = env._reward_stance_grf()

    torch.testing.assert_close(swing, torch.tensor([-1.0, -1.0, 0.0, 0.0]))
    torch.testing.assert_close(stance, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    expected = -(env.grf * env.oscillators.sin()).sum(dim=1)
    torch.testing.assert_close(swing + stance, expected)


def test_cursorial_reward_uses_physical_hip_angles_only(env):
    env.dof_pos.fill_(5.0)
    hips = [i for i, name in enumerate(env.dof_names) if name.endswith("_hip_joint")]
    env.dof_pos[:, hips] = 0.0
    torch.testing.assert_close(env._reward_cursorial(), torch.zeros(env.num_envs))

    env.dof_pos[:, hips] = env.scales["dof_pos"][hips] * 0.4
    torch.testing.assert_close(
        env._reward_cursorial(), torch.full((env.num_envs,), -0.16)
    )

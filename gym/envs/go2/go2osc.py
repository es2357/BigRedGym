import math
import torch

from gym.envs.go2.go2 import Go2
from gym.utils.sampling import masked_update


class Go2Osc(Go2):
    """ORC(111) oscillator dynamics and phase rewards for Go2

        for mental sanity: 
            paper leg naming convention vs code (from URDF): 
            "RF" = "FR", "LF" = "FL", "RH" = "RR", "LH" = "RL"

        comments : translate code to math in paper 
    """

    # inherited order from LeggedRobot
    LEG_NAMES = ("FL", "FR", "RL", "RR")

    def _init_buffers(self):
        super()._init_buffers()
        
        self._gravity_magnitude = 9.81  # m/s^2

        # phi : each leg's phase angle [rad]
        self.oscillators = torch.zeros(self.num_envs, 4, device=self.device)
        # phi dot [rad/s]
        self.oscillators_vel = torch.zeros_like(self.oscillators)
        # cos(phi) and sin(phi) for each of four legs 
        self.oscillator_obs = torch.zeros(self.num_envs, 8, device=self.device)
        self.grf = torch.zeros_like(self.oscillators)
        # nominal frequency [Hz]
        self.osc_omega = torch.full(
            (self.num_envs, 1),
            self.cfg.osc.omega_stop,
            device=self.device,
            dtype=self.oscillators.dtype,
        )
        # sigma -- strength of force feedback
        self.osc_coupling = torch.full_like(self.osc_omega, self.cfg.osc.coupling_stop)
        # xi -- offset in feedback term
        self.osc_offset = torch.full_like(self.osc_omega, self.cfg.osc.offset)

        hip_names = [f"{leg}_hip_joint" for leg in self.LEG_NAMES]
        self._hip_indices = torch.tensor(
            self.robot_layout.dof_indices(hip_names),
            device=self.device,
            dtype=torch.long,
        )
        self._update_oscillator_observations()


    def _reset_idx(self, reset_mask):
        if not reset_mask.any():
            return
        super()._reset_idx(reset_mask)
        # The PD controller adds the nominal pose to these position offsets
        self.dof_pos_target.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self.dof_pos_history.masked_fill_(reset_mask.unsqueeze(1), 0.0)

    def _reset_system(self, reset_mask):
        super()._reset_system(reset_mask)
        self._reset_oscillators(reset_mask)

    def _reset_oscillators(self, reset_mask):
        phases = torch.rand_like(self.oscillators) * (2 * math.pi)
        masked_update(self.oscillators, phases, reset_mask)
        # Reset contacts may be stale until the backend takes a physics step
        self.grf.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        masked_update(self.oscillators_vel, self._compute_phase_velocity(), reset_mask)
        self._update_oscillator_observations(reset_mask)

    def _compute_grf(self, grf_norm=True):
        """Return RF, LF, RH, LH force magnitudes, optionally divided by weight."""
        forces = self.contact_forces[:, self.feet_indices, :]
        magnitudes = torch.linalg.vector_norm(forces, dim=-1)
        if not grf_norm:
            return magnitudes
        # Read applied masses here: startup randomization runs after buffer setup.
        weight = self._backend.link_mass.sum(dim=1, keepdim=True)
        weight = weight * self._gravity_magnitude
        return (magnitudes / weight).clamp(0.0, 1.0)

    def _update_oscillator_parameters(self, reset_mask=None):
        """Apply the command-dependent frequency and feedback schedule, Eq. (5)."""
        osc = self.cfg.osc
        speed = self.commands[:, :1].abs()
        stopped = speed <= osc.stop_threshold
        moving_speed = (speed - osc.stop_threshold).clamp_min(0.0)
        omega = torch.where(
            stopped,
            osc.omega_stop,
            (osc.omega_step + osc.omega_slope * moving_speed).clamp(0.0, osc.omega_max),
        )
        coupling = torch.where(
            stopped,
            osc.coupling_stop,
            (osc.coupling_step + osc.coupling_slope * moving_speed).clamp(
                0.0, osc.coupling_max
            ),
        )
        offset = stopped.to(self.oscillators.dtype) * osc.offset
        if osc.omega_var > 0:
            omega = omega + torch.randn_like(omega) * osc.omega_var
        if osc.coupling_var > 0:
            coupling = coupling + torch.randn_like(coupling) * osc.coupling_var
        omega = omega.clamp_min(0.1)
        coupling = coupling.clamp_min(0.0)

        for target, value in (
            (self.osc_omega, omega),
            (self.osc_coupling, coupling),
            (self.osc_offset, offset),
        ):
            if reset_mask is None:
                target.copy_(value)
            else:
                masked_update(target, value, reset_mask)

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self._update_oscillator_parameters()

    def _post_physics_step(self):
        super()._post_physics_step()
        self._step_oscillators(self.dt / self.cfg.control.decimation)

    def _compute_phase_velocity(self):
        feedback = (
            self.osc_coupling
            * self.grf
            * (torch.cos(self.oscillators) + self.osc_offset)
        )
        return 2 * math.pi * (self.osc_omega - feedback)

    def _step_oscillators(self, dt=None):
        """Advance Eq. (4) with fresh foot forces at the physics timestep."""
        if dt is None:
            dt = self.dt / self.cfg.control.decimation
        self.grf.copy_(self._compute_grf())
        velocity = self._compute_phase_velocity()
        if self.cfg.osc.process_noise_std > 0:
            # Reference process noise is in cycles/s, before conversion to rad/s.
            velocity += (
                2
                * math.pi
                * self.cfg.osc.process_noise_std
                * torch.randn_like(velocity)
            )
        self.oscillators_vel.copy_(velocity)
        self.oscillators.add_(velocity * dt).remainder_(2 * math.pi)
        self._update_oscillator_observations()

    def _update_oscillator_observations(self, reset_mask=None):
        observations = torch.cat(
            (torch.cos(self.oscillators), torch.sin(self.oscillators)), dim=1
        )
        if reset_mask is None:
            self.oscillator_obs.copy_(observations)
        else:
            masked_update(self.oscillator_obs, observations, reset_mask)

    def _resample_commands(self, command_mask):
        """Sample reference-style command centers, noise, and standing cases."""
        if not command_mask.any():
            return
        candidate = torch.empty_like(self.commands)
        centers = torch.tensor(
            self.command_ranges["lin_vel_x"], device=self.device, dtype=torch.float
        )
        indices = torch.randint(centers.numel(), (self.num_envs,), device=self.device)
        candidate[:, 0] = (
            centers[indices] + torch.randn_like(candidate[:, 0]) * self.cfg.commands.var
        )
        candidate[:, 1] = (
            2 * torch.rand(self.num_envs, device=self.device) - 1
        ) * self.command_ranges["lin_vel_y"]
        candidate[:, 2] = (
            2 * torch.rand(self.num_envs, device=self.device) - 1
        ) * self.command_ranges["yaw_vel"]

        if 0.0 in self.command_ranges["lin_vel_x"]:
            # Independently include forward-only, rotation-only, and zero commands.
            candidate[:, 1:] *= torch.rand(self.num_envs, 1, device=self.device) < 0.8
            candidate[:, :2] *= torch.rand(self.num_envs, 1, device=self.device) < 0.8
            candidate *= torch.rand(self.num_envs, 1, device=self.device) < 0.9
        masked_update(self.commands, candidate, command_mask)
        self._update_oscillator_parameters(command_mask)
        masked_update(
            self.oscillators_vel, self._compute_phase_velocity(), command_mask
        )

    def _reward_swing_grf(self):
        """Penalize loading during positive-sine swing."""
        swing = torch.sin(self.oscillators).clamp_min(0.0)
        return -(self.grf * swing).sum(dim=1)

    def _reward_stance_grf(self):
        """Reward loading during negative-sine stance."""
        stance = (-torch.sin(self.oscillators)).clamp_min(0.0)
        return (self.grf * stance).sum(dim=1)

    def _reward_cursorial(self):
        """Penalize hip abduction/adduction away from zero."""
        hip_positions = self.get_state("dof_pos").index_select(1, self._hip_indices)
        return -hip_positions.square().mean(dim=1)

    def _reward_action_rate(self):
        """Match the reference first-difference penalty."""
        n = self.num_actuators
        change = (
            self.dof_pos_history[:, :n]
            - self.dof_pos_history[:, n:2 * n]
        )
        denominator = (self.dt * self.cfg.control.decimation) ** 2
        return -change.square().sum(dim=1) / denominator

    def _reward_action_rate2(self):
        """Match the reference second-difference penalty."""
        n = self.num_actuators
        change = (
            self.dof_pos_history[:, :n]
            - 2 * self.dof_pos_history[:, n:2 * n]
            + self.dof_pos_history[:, 2 * n:]
        )
        denominator = (self.dt * self.cfg.control.decimation) ** 2
        return -change.square().sum(dim=1) / denominator

"""ORC(111) configuration adapted to Go2 and BigRedGym's PPO2 runner"""

from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2OscCfg(Go2Cfg):
    class env(Go2Cfg.env):
        # num_envs = 4096
        episode_length_s = 10.0

    class init_state(Go2Cfg.init_state):
        # Begin from the nominal Go2 pose at rest.
        # reset_mode = "reset_to_basic"
        # lin_vel = [0.0, 0.0, 0.0]
        # ang_vel = [0.0, 0.0, 0.0]

        # Policy targets are offsets from this static nominal posture.
        default_joint_angles = {
            "hip_joint": 0.0,
            "calf_joint": -1.36,
            "FL_thigh_joint": 0.96,
            "FR_thigh_joint": 0.96,
            "RL_thigh_joint": 0.96,
            "RR_thigh_joint": 0.96,
        }
        reset_mode = "reset_to_basic"

        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.450, 0.50],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]
        root_vel_range = [
            [-0.5, 3.0],  # x
            [-0.1, 0.1],  # y
            [-0.05, 0.05],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]

    class control(Go2Cfg.control):
        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 20.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 0.5}
        ctrl_frequency = 50
        desired_sim_frequency = 300

    class osc:
        # Constants related to oscillator dynamics
        stop_threshold = 0.5
        omega_stop = 1.0
        omega_step = 1.0  # 1.5; set lower to account for a larger, heavier body
        omega_slope = 0.5  # 1.0
        omega_max = 4.0
        coupling_stop = 4.0
        coupling_step = 1.0
        coupling_slope = 0.0  # unused
        coupling_max = 1.0  # unused
        offset = 1.0

        # Reference standard deviations; set all three to zero for diagnostics.
        omega_var = 0.25
        coupling_var = 0.25
        process_noise_std = 0.25

    class commands(Go2Cfg.commands):
        resampling_time = 3.0
        var = 0.1

        class ranges(Go2Cfg.commands.ranges):
            # lin_vel_x = [1.0]
            # lin_vel_y = 0.0
            # yaw_vel = 0.0
            lin_vel_x = [-0.5, 0.0, 0.5, 1.0, 2.0]
            lin_vel_y = 0.5
            yaw_vel = 1.5

    class push_robots(Go2Cfg.push_robots):
        toggle = True
        interval_s = 5
        # max_push_vel_xy = 0.2

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            # contact_friction_range = None
            # link_mass_scale_range = None
            link_mass_scale_range = [0.9, 1.2]

        class episode(Go2Cfg.domain_randomization.episode):
            scale_ranges = {}  # Keep the nominal PD gains fixed.

    class scaling(Go2Cfg.scaling):
        base_ang_vel = 0.3#[0.3, 0.3, 0.1]
        # Canonical joint order: FL, FR, RL, RR; hip, thigh, calf per leg.
        dof_pos = 4 * [0.2, 0.3, 0.3]
        dof_pos_obs = dof_pos
        dof_pos_target = dof_pos


class Go2OscRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        hidden_dims = [256, 256, 128]
        activation = "elu"
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "oscillator_obs",
            "dof_pos_target",
        ]
        normalize_obs = False
        smooth_exploration = False
        actions = ["dof_pos_target"]
        add_noise = False
        disable_actions = False

        class noise:
            # The runner applies these directly; phase features stay exact.
            base_ang_vel = 0.01
            dof_pos_obs = 0.01
            dof_vel = 0.005

    class critic(Go2RunnerCfg.critic):
        # hidden_dims = [256, 256, 128]
        hidden_dims = [128, 64]
        activation = "elu"
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "oscillator_obs",
            "oscillators_vel",
            "dof_pos_target",
        ]
        normalize_obs = False

        class reward(Go2RunnerCfg.critic.reward):
            class weights(Go2RunnerCfg.critic.reward.weights):
                tracking_lin_vel = 4.0
                tracking_ang_vel = 4.0
                lin_vel_z = 0.0
                ang_vel_xy = 0.2
                orientation = 1.0
                torques = 12 * 5.0e-7  # Local torque reward uses a joint mean.
                # Keep Go2 smoothing: its formulas differ from ORCAgym's.
                action_rate = 0.01
                action_rate2 = 0.001
                swing_grf = 6.0
                stance_grf = 2.0
                cursorial = 0.5
                dof_near_home = 0.15  # to stop legs from splaying out
                even_load = 4.0  # to stop one leg from being unused or dragged
                feet_contact_forces = 0.5  # try to reduce huge impacts; might also reduce bounding/hopping behavior
                torque_limits = 1.0  # we absolutely need to impose torque limits
                dof_vel_limits = 1.0  # we also need to impose limits on dof velocity
                dof_pos_limits = 1.0  # maybe this will help keep limbs away from the edges of their extents
                # TODO: adding the limit rewards really messed up the policy; now squats right above the ground with very tight angles

            class termination_weight(Go2RunnerCfg.critic.reward.termination_weight):
                termination = 0.15

    class algorithm(Go2RunnerCfg.algorithm):
        # Reference collection size; four passes of eight minibatches per update.
        rollout_size = 4096 * 32
        batch_size = rollout_size // 8
        max_gradient_steps = 4 * 8
        clip_param = 0.2
        entropy_coef = 0.02
        learning_rate = 1.0e-5
        schedule = "adaptive"
        desired_kl = 0.01
        lr_range = [1.0e-5, 3.0e-4]
        lr_ratio = 1.5
        use_clipped_value_loss = True
        max_grad_norm = 1.0
        discount_horizon = 1.0
        GAE_bootstrap_horizon = 2.0

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2_osc"
        run_name = "ORC111"
        max_iterations = 2000
        save_interval = 25
        algorithm_class_name = "PPO2"
        resume = False

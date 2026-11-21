"""Evaluate the effect of changing P and D gains (stiffness and damping) on a policy
"""

import argparse
import random
import sys
import time

import torch
import matplotlib.pyplot as plt

from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry


def get_measure_args(argv=None):
    parser = argparse.ArgumentParser(description="Measure a Go2 policy's response to changing stiffness and damping")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min_kp", type=float, default=0.0)
    parser.add_argument("--max_kp", type=float, default=50.0)
    parser.add_argument("--min_kd", type=float, default=0.0)
    parser.add_argument("--max_kd", type=float, default=5.0)
    parser.add_argument("--num_kp", type=int, default=25)
    parser.add_argument("--num_kd", type=int, default=50)
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Checkpoint root under logs/ (default: task config value)",
    )
    parser.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Run directory name under logs/<experiment_name>/ (default: latest)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="Model iteration index (default: latest)",
    )
    parser.add_argument(
        "--original_cfg",
        action="store_true",
        help="Load environment and runner configs saved with the selected run.",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--backend",
        type=str,
        default="mujoco",
        choices=["mujoco", "vsim"],
        help="Physics backend (vsim is CUDA-only; start via "
        "`uv run --env-file .env.vsim ...`)",
    )
    return parser.parse_args(argv)


def setup(args):
    # Register tasks (imports the task classes).
    import gym.envs  # noqa: F401

    env_cfg, train_cfg = task_registry.get_cfgs(
        args.task,
        original_cfg=args.original_cfg,
        experiment_name=args.experiment_name,
        load_run=args.load_run,
    )

    # Play-time overrides — small batch, long episodes, no pushes.
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = 1  # changed from 50
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 9999
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    env_cfg.init_state.reset_mode = "reset_to_basic"
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.show_ui = args.viewer_ui

    if args.seed is not None:
        env_cfg.seed = args.seed
        train_cfg.seed = args.seed
    else:
        env_cfg.seed = random.randint(0, 10000)
        train_cfg.seed = env_cfg.seed

    train_cfg.runner.device = args.device
    train_cfg.runner.resume = True
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    train_cfg.logging.enable_local_saving = False

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    set_seed(env_cfg.seed)

    env = task_registry.make_env(
        args.task,
        env_cfg,
        device=args.device,
        headless=args.headless,
        backend=args.backend,
    )

    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return env, runner


def reset_gains(env):
    # Reset the stiffness and damping (p_gains and d_gains) manually
    for actuator_index, name in enumerate(env.actuated_dof_names):
        matching_gains = [
            gain_name
            for gain_name in env.cfg.control.stiffness
            if gain_name in name
        ]
        if len(matching_gains) != 1:
            raise ValueError(
                f"expected exactly one PD gain match for {name!r}, "
                f"got {matching_gains}"
            )
        gain_name = matching_gains[0]
        env.p_gains[:, actuator_index] = env.cfg.control.stiffness[gain_name]
        env.d_gains[:, actuator_index] = env.cfg.control.damping[gain_name]


def play(env, runner):
    n_steps = 10 * int(env.max_episode_length)
    # Capture the starting positions of the envs
    start_root_states = env.root_states[:,:2].clone()
    sum_tracking_reward = torch.zeros_like(env.root_states[:,0])
    for step in range(n_steps):
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        sum_tracking_reward += env._reward_tracking_lin_vel()
        if env.exit:
            break
    # Record the distance traveled and whether the robot is upright
    # Need to offset for initial position (not always 0,0,0)
    displacement = torch.sum(env.commands[:,:2] * (env.root_states[:,:2] - start_root_states), axis=1)  # num envs x 2 --> root planar position 2-vectors
    upright = torch.sum(env.gravity_vec * env.projected_gravity, axis=1)  # 1 if upright, 0 if sideways, -1 if flipped
    return displacement, upright, sum_tracking_reward


def format_seconds_to_hms(secs):
    minutes, seconds = divmod(int(secs), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    print("Setup...")
    script_args = get_measure_args()

    stiffness = torch.linspace(script_args.min_kp, script_args.max_kp, script_args.num_kp)
    damping = torch.linspace(script_args.min_kd, script_args.max_kd, script_args.num_kd)

    class args:
        task = script_args.task
        device = script_args.device
        num_envs = script_args.num_envs
        seed = script_args.seed
        experiment_name = script_args.experiment_name
        load_run = script_args.load_run
        checkpoint = script_args.checkpoint
        original_cfg = script_args.original_cfg
        headless = True
        backend = script_args.backend
        viewer_ui = False
        keyboard = False

    print("Beginning sweep...")
    # Namespace(task='go2trot', device='cpu', num_envs=1, seed=None, experiment_name=None, load_run=None, checkpoint=-1, original_cfg=False, headless=False, backend='mujoco', viewer_ui=False, keyboard=True)
    displacement_means = torch.zeros((stiffness.size()[0], damping.size()[0]))
    displacement_stds = torch.zeros((stiffness.size()[0], damping.size()[0]))
    upright_means = torch.zeros((stiffness.size()[0], damping.size()[0]))
    upright_stds = torch.zeros((stiffness.size()[0], damping.size()[0]))
    sum_forward_reward_means = torch.zeros((stiffness.size()[0], damping.size()[0]))
    sum_forward_reward_stds = torch.zeros((stiffness.size()[0], damping.size()[0]))
    with torch.no_grad():
        env, runner = setup(args)
        # Save the original stiffness and damping values to plot
        # This assumes that stiffness and damping is uniform across all joints
        orig_stiffness = list(env.cfg.control.stiffness.values())[0]
        orig_damping = list(env.cfg.control.damping.values())[0]
        # Starting the start time way before the sweep causes a big offset error
        # So we will start it immediately before
        start_time = time.time()
        for p_idx in range(stiffness.size()[0]):
            for d_idx in range(damping.size()[0]):
                now_time = time.time()
                percent_complete = (p_idx * damping.size()[0] + d_idx) / (stiffness.size()[0] * damping.size()[0])
                if percent_complete != 0:
                    print(f"Percent: {100 * percent_complete :.1f}% (Testing k={stiffness[p_idx]}, b={damping[d_idx]}); Elapsed: {format_seconds_to_hms(now_time - start_time)}; Estimated time remaining: {format_seconds_to_hms((now_time - start_time) / percent_complete - (now_time - start_time))} (Estimated total {format_seconds_to_hms((now_time - start_time) / percent_complete)})")
                env.reset()
                # Start with forward command
                env.commands[:] = 0.0
                env.commands[:, 0] = 1.5  # mid-range command value
                # Reset the stiffness and damping
                # Referencing the keys of the dicts allows this to work for any robot
                for key in env.cfg.control.stiffness.keys():
                    env.cfg.control.stiffness[key] = stiffness[p_idx]
                    env.cfg.control.damping[key] = damping[d_idx]
                reset_gains(env)

                displacement, upright, sum_forward_reward = play(env, runner)
                displacement_means[p_idx, d_idx] = torch.mean(displacement)
                displacement_stds[p_idx, d_idx] = torch.std(displacement)
                upright_means[p_idx, d_idx] = torch.mean(upright)
                upright_stds[p_idx, d_idx] = torch.std(upright)
                sum_forward_reward_means[p_idx, d_idx] = torch.mean(sum_forward_reward)
                sum_forward_reward_stds[p_idx, d_idx] = torch.std(sum_forward_reward)

    class DatasetPackage:
        def __init__(self, data, title, cmap_label, cmap_cmap, filename):
            self.data = data
            self.title = title
            self.cmap_label = cmap_label
            self.cmap_cmap = cmap_cmap
            self.filename = filename

    # Package up the data and some descriptions
    datasets = [
        DatasetPackage(displacement_means, "Displacement products means", "Bigger positive = displacement matches commands", "viridis", "displacement_mean.png"),
        DatasetPackage(displacement_stds, "Displacement product stds", "Bigger value = more variation", "viridis", "displacement_std.png"),
        DatasetPackage(upright_means, "Upright products means", "1 = upright, 0 = sideways, -1 = upside-down", "magma", "upright_mean.png"),
        DatasetPackage(upright_stds, "Upright products stds", "Bigger value = more variation", "magma", "upright_std.png"),
        DatasetPackage(sum_forward_reward_means, "Sum of forward rewards means", "Bigger positive = forward velocity tracked better", "cividis", "sum_forward_reward_mean.png"),
        DatasetPackage(sum_forward_reward_stds, "Sum of forward rewards stds", "Bigger value = more variation", "cividis", "sum_forward_reward_std.png")
    ]

    # Display the arrays as an image
    for dataset in datasets:
        plt.clf()
        plt.imshow(torch.flip(dataset.data, [0]), cmap=dataset.cmap_cmap, interpolation='nearest', aspect='auto',
                   extent =[torch.min(damping).item(), torch.max(damping).item(), torch.min(stiffness).item(),
                            torch.max(stiffness).item()])
        cbar = plt.colorbar()
        cbar.set_label(dataset.cmap_label, rotation=270, labelpad=15)
        plt.plot(orig_damping, orig_stiffness, 'o', markerfacecolor='none', markeredgecolor='red', label="Trained")
        plt.xlabel("Damping (derivative gain)")
        plt.ylabel("Stiffness (proportional gain)")
        plt.title(f"{dataset.title}\nEnvs: {script_args.num_envs}; Sim / Control Freq Ratio: {env.cfg.control.desired_sim_frequency / env.cfg.control.ctrl_frequency}")
        plt.legend()
        # plt.show()
        plt.savefig(dataset.filename)


    # On cpu with stiffness 0:5:50, damping 0:0.1:1, desired frequency 150:
    # 5 second episodes, 1 env: 16.3 seconds per 1%
    # 5 second episodes, 2 envs: 23.5 seconds per 1%
    # 5 second episodes, 4 envs: 30 seconds per 1%
    # 5 second episodes, 8 envs: 54 seconds per 1%
    # --> Time is roughly linear with the number of envs, seconds per % = 5.254 envs + 11.24783
    # 1 second episodes, 4 env: 6.4 seconds per 1%
    # 1 second episodes, 8 env: 11 seconds per 1%


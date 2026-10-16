from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from scripts.play import get_play_args, setup

import argparse

from gym.utils.interfaces.MujocoKeyboardInterface import MujocoKeyboardInterface
from gym.utils.interfaces.CommandVisualizer import CommandVisualizer


def main():
    # Read our plotting option, then forward the other options to play.py.
    mode_parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )
    mode_parser.add_argument("--live", action="store_true")
    mode_args, remaining = mode_parser.parse_known_args()

    args = get_play_args(remaining)
    args.live = mode_args.live
    args.num_envs = 1
    args.headless = not args.live

    if args.live and (
        args.device != "cpu" or args.backend != "mujoco"
    ):
        mode_parser.error(
            "--live requires --device=cpu --backend=mujoco"
        )

    rows = []

    with torch.no_grad():
        env, runner = setup(args)

        try:
            env.commands.zero_()
            env.cfg.commands.resampling_time = 1e9

            if args.live:
                MujocoKeyboardInterface(env)
                CommandVisualizer(env)
                print(
                    "Live recording: use Up/Down. "
                    "Close the viewer or press Ctrl+C to save."
                )

            cfg = env.cfg.height_control
            targets = [
                cfg.default_height,
                cfg.min_height,
                cfg.max_height,
                cfg.default_height,
            ]
            steps_per_target = round(5.0 / env.dt)
            start_step = env.common_step_counter
            step = 0

            def record_sample():
                rows.append(
                    (
                        (env.common_step_counter - start_step) * env.dt,
                        env.height_command[0, 0].item(),
                        env.base_height[0, 0].item(),
                    )
                )

            while not env.exit:
                if args.live:
                    viewer = env._backend._viewer
                    if viewer is not None and not viewer.is_running():
                        break
                else:
                    if step >= len(targets) * steps_per_target:
                        break

                    target = targets[step // steps_per_target]
                    env.height_command.fill_(target)

                record_sample()

                runner.set_actions(
                    runner.actor_cfg["actions"],
                    runner.get_inference_actions(),
                    runner.actor_cfg["disable_actions"],
                )
                env.step()
                step += 1

            record_sample()

        except KeyboardInterrupt:
            print("\nRecording stopped.")

        finally:
            viewer = getattr(env._backend, "_viewer", None)
            if viewer is not None:
                viewer.close()
            env._backend.close()

    if not rows:
        print("No height samples were recorded.")
        return

    # Your existing data, CSV, and plotting code continues here.
    data = np.asarray(rows)
    time_s = data[:, 0]
    commanded = data[:, 1]
    actual = data[:, 2]

    output = Path("logs/height_tracking")
    output.mkdir(parents=True, exist_ok=True)

    mode = "live" if args.live else "auto"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = output / (
        f"{args.experiment_name}_checkpoint_{runner.it}_{mode}_{stamp}"
    )
    np.savetxt(
        stem.with_suffix(".csv"),
        data,
        delimiter=",",
        header="time_s,commanded_height_m,actual_height_m",
        comments="",
        fmt="%.6f",
    )

    fig = Figure(figsize=(9, 4))
    ax = fig.subplots()

    ax.step(
        time_s,
        commanded,
        where="post",
        linestyle="--",
        color="black",
        label="Commanded height",
    )
    ax.plot(time_s, actual, label="Actual height", color="tab:blue")

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Body height (m)")
    ax.set_title("Height Tracking")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300)

    print(f"Graph: {stem.with_suffix('.png')}")
    print(f"Data:  {stem.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

"""Record a following-camera Go2 video with gait labels. Requires ffmpeg."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from collections import Counter
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from gym import GYM_ROOT_DIR
from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.utils.gait_classifier import GaitClassifier, GaitResult

if __package__:
    from .classify_gait import FOOT_NAMES
    from .play import get_play_parser, setup
else:
    from classify_gait import FOOT_NAMES
    from play import get_play_parser, setup


def get_args(argv=None):
    parser = get_play_parser()
    parser.description = "Record a Go2 policy with a following camera and gait labels"
    parser.set_defaults(
        num_envs=1,
        device="cpu",
        backend="mujoco",
        headless=True,
        keyboard=False,
        seed=123,
        original_cfg=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="Go2")
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--contact_on", type=float, default=5)
    parser.add_argument("--contact_off", type=float, default=2)
    args = parser.parse_args(argv)
    if args.device != "cpu" or args.backend != "mujoco" or args.num_envs != 1:
        parser.error("recording requires CPU MuJoCo and one environment")
    if not 0 < args.duration < 50 or args.fps <= 0:
        parser.error("duration must be between 0 and 50 seconds; fps must be positive")
    if min(args.width, args.height) < 240 or args.width % 2 or args.height % 2:
        parser.error("video dimensions must be even and at least 240 pixels")
    if not 0 <= args.contact_off < args.contact_on < math.inf:
        parser.error("contact cutoffs must satisfy 0 <= off < on < infinity")
    if round(args.duration * args.fps) < 1:
        parser.error("duration must produce at least one frame")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.headless = True
    return args


def overlay(frame, label, result, time_s, fonts):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    title, detail = fonts
    gait = f"Gait: {result.label}"
    if result.valid:
        description = f"Distance: {result.distance:.2f} / 2.00"
        if result.label == "Transition":
            description += f" | nearest: {result.nearest_gait}"
    else:
        description = result.reason or f"Completed cycles: {result.cycles_used} / 2"
    text_width = max(
        draw.textbbox((0, 0), gait, font=title)[2],
        draw.textbbox((0, 0), description, font=detail)[2],
    )
    width = min(image.width - 36, max(380, text_width + 32))
    draw.rounded_rectangle((18, 18, 18 + width, 116), radius=10, fill=(12, 22, 34, 225))
    color = (
        (146, 239, 177)
        if result.valid and result.label != "Transition"
        else (255, 213, 131)
    )
    draw.text((34, 28), gait, font=title, fill=color)
    draw.text((34, 77), description, font=detail, fill="white")
    draw.rounded_rectangle(
        (18, image.height - 55, image.width - 18, image.height - 14),
        radius=8,
        fill=(12, 22, 34, 220),
    )
    draw.text((32, image.height - 48), label, font=detail, fill="white")
    draw.text(
        (image.width - 110, image.height - 48),
        f"{time_s:5.2f} s",
        font=detail,
        fill="white",
    )
    return np.asarray(image)


@torch.no_grad()
def record(env, runner, args):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the video")
    visual_path = env.cfg.asset.mujoco_visual_model.format(GYM_ROOT_DIR=GYM_ROOT_DIR)
    model = MuJocoCPUBackend()._load_model(env.cfg, visual_model_path=visual_path)
    sim_model, sim_data = env._backend._mjm, env._backend._datas[0]
    if model.nq != sim_model.nq or model.nv != sim_model.nv:
        raise ValueError("visual and simulation models have different state layouts")
    if tuple(model.joint(i).name for i in range(model.njnt)) != tuple(
        sim_model.joint(i).name for i in range(sim_model.njnt)
    ):
        raise ValueError("visual and simulation joint orders differ")
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 2.0, 150, -20
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    fonts = (
        ImageFont.truetype("DejaVuSans-Bold.ttf", 28),
        ImageFont.truetype("DejaVuSans.ttf", 19),
    )
    feet = torch.tensor(env.robot_layout.body_indices(FOOT_NAMES), dtype=torch.long)
    classifier = GaitClassifier(
        contact_on=args.contact_on, contact_off=args.contact_off
    )
    rows, frame_times, frame_gaits = [], [], []
    failure_s = None
    completed_steps = 0

    def sample():
        nonlocal failure_s
        time_s = completed_steps * env.dt
        forces = env.contact_forces[0, feet].norm(dim=-1).numpy()
        if failure_s is None and bool(env.terminated[0] | env.timed_out[0]):
            failure_s = time_s
            classifier.reset()
        if failure_s is None:
            result = classifier.update(time_s, forces)
        else:
            result = GaitResult("Episode ended", reason="Termination detected")
        row = {"time_s": time_s, **dict(zip(FOOT_NAMES, forces.tolist()))}
        row.update(
            gait=result.label,
            nearest=result.nearest_gait,
            distance=result.distance,
            vx_m_s=float(env.base_lin_vel[0, 0]),
        )
        rows.append(row)
        return result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{args.width}x{args.height}",
            "-r",
            str(args.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    frame_count = round(args.duration * args.fps)
    try:
        result = sample()
        for frame_index in range(frame_count):
            target_step = math.ceil(frame_index / (args.fps * env.dt) - 1e-9)
            while completed_steps < target_step:
                runner.set_actions(
                    runner.actor_cfg["actions"],
                    runner.get_inference_actions(),
                    runner.actor_cfg["disable_actions"],
                )
                env.step()
                completed_steps += 1
                result = sample()

            # Keep rendering separate so mj_forward cannot affect the simulation.
            data.qpos[:] = sim_data.qpos
            data.qvel[:] = sim_data.qvel
            mujoco.mj_forward(model, data)
            camera.lookat[:] = (data.qpos[0], data.qpos[1], 0.35)
            renderer.update_scene(data, camera=camera)
            time_s = completed_steps * env.dt
            frame = overlay(renderer.render(), args.label, result, time_s, fonts)
            encoder.stdin.write(frame.tobytes())
            frame_times.append(time_s)
            frame_gaits.append(result.label)
            if frame_index == frame_count // 2:
                Image.fromarray(frame).save(args.output.with_suffix(".png"))
            if frame_index % args.fps == 0:
                print(f"{args.label}: {time_s:.2f}s | {result.label}", flush=True)
    finally:
        renderer.close()
        encoder.stdin.close()
        error = encoder.stderr.read().decode()
        if encoder.wait():
            raise RuntimeError(f"ffmpeg failed: {error}")

    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "run": args.load_run,
        "checkpoint": runner.it,
        "seed": args.seed,
        "backend": type(env._backend).__name__,
        "duration_s": frame_count / args.fps,
        "fps": args.fps,
        "frames": frame_count,
        "command": env.commands[0].tolist(),
        "contact_on_N": args.contact_on,
        "contact_off_N": args.contact_off,
        "gait_distance_threshold": 2.0,
        "sampling_hz": 1 / env.dt,
        "first_failure_s": failure_s,
        "frame_times_s": frame_times,
        "gait_frame_counts": dict(Counter(frame_gaits)),
        "last_classification": asdict(result),
        "camera": {
            "mode": "following",
            "distance": 2.0,
            "azimuth": 150,
            "elevation": -20,
        },
        "video": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


def main(argv=None):
    args = get_args(argv)
    env, runner = setup(args)
    try:
        record(env, runner, args)
    finally:
        env._backend.close()


if __name__ == "__main__":
    main()

"""Play an RSL-RL checkpoint while injecting custom commands into the environment.

This is a lightweight variant of `play.py` that sets `env.unwrapped.vel_commands` and
`env.unwrapped._z_command` before each policy invocation so you can evaluate the
policy under your own commanded velocity/height trajectories.

Usage examples:
  python play_cmds.py --task "Go2-Play" --vx 0.5 --vy 0.0 --wz 0.0 --z 0.35
  python play_cmds.py --task "Go2-Play" --cmd_wave --vx 0.5 --z 0.35

Launch Isaac Sim before running this script (same as `play.py`).
"""

import argparse
import sys
from isaaclab.app import AppLauncher
import cli_args  # isort: skip




# local imports

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL checkpoint while injecting commands.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--vx", type=float, default=0.0, help="X linear velocity command to inject.")
parser.add_argument("--vy", type=float, default=0.0, help="Y linear velocity command to inject.")
parser.add_argument("--wz", type=float, default=0.0, help="Yaw angular velocity command to inject.")
parser.add_argument("--z", type=float, default=0.35, help="Z target command to inject.")
parser.add_argument("--cmd_wave", action="store_true", default=False, help="Modulate commands with a slow sinusoid.")
parser.add_argument("--disable_resampler", action="store_true", default=False, help="If the env has a resampler, disable it during play.")
parser.add_argument(
    "--keyboard-layout",
    type=str,
    choices=["qwerty", "azerty"],
    default="azerty",
    help="Keyboard layout to interpret alternative letter keys (affects WASD-like mappings).",
)
parser.add_argument(
    "--stdin-teleop",
    action="store_true",
    default=True,
    help="Use terminal stdin for teleop (useful over RDP).",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
import time
import torch
import gymnasium as gym
import math

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

"""
To run:
python scripts/reinforcement_learning/rsl_rl/play_cmds.py --task Isaac-Velocity-Go2-Direct-v0 --vx 0 --vy 0 --wz 0 --z 0.3 --num_envs 1
"""

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent while injecting commands into the env."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # update agent/env cfgs with CLI overrides
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # recover checkpoint path
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_cmds"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract NN module for export if desired
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # normalizer (if present)
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy artifacts
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # build command tensors (applied to all envs)
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    base_vx = args_cli.vx
    base_vy = args_cli.vy
    base_wz = args_cli.wz
    base_z = args_cli.z

    # -------------------- keyboard control setup --------------------
    # shared command state updated by keyboard listener
    import threading

    cmd_state = {
        "vx": float(base_vx),
        "vy": float(base_vy),
        "wz": float(base_wz),
        "z": float(base_z),
    }
    cmd_lock = threading.Lock()

    # step sizes for each keypress
    V_STEP = 0.05
    W_STEP = 0.05
    Z_STEP = 0.02

    # try to import pynput for keyboard listening; fallback gracefully
    try:
        from pynput import keyboard

        def _on_press(key):
            # handle character keys and special keys (arrows)
            char = None
            try:
                char = key.char.lower()
            except Exception:
                pass
            with cmd_lock:
                # character mappings (alternative controls)
                # support QWERTY (default) and AZERTY layouts
                layout = args_cli.keyboard_layout if hasattr(args_cli, "keyboard_layout") else "qwerty"

                # normalize AZERTY differences: on AZERTY keyboards, 'z' <-> 'w' and 'q' <-> 'a'
                def is_key(k):
                    return char == k if char is not None else False

                if layout == "qwerty":
                    if is_key('w'):
                        cmd_state['vx'] += V_STEP
                    elif is_key('s'):
                        cmd_state['vx'] -= V_STEP
                    elif is_key('a'):
                        cmd_state['vy'] += V_STEP
                    elif is_key('d'):
                        cmd_state['vy'] -= V_STEP
                    elif is_key('q'):
                        cmd_state['wz'] += W_STEP
                    elif is_key('e'):
                        cmd_state['wz'] -= W_STEP
                    elif is_key('r'):
                        cmd_state['z'] += Z_STEP
                    elif is_key('f'):
                        cmd_state['z'] -= Z_STEP
                    elif is_key('z'):
                        # reset to base
                        cmd_state['vx'] = float(base_vx)
                        cmd_state['vy'] = float(base_vy)
                        cmd_state['wz'] = float(base_wz)
                        cmd_state['z'] = float(base_z)
                else:  # azerty
                    # on AZERTY: 'z' behaves like 'w', 's' still down, 'q' behaves like 'a', 'd' same
                    if is_key('z'):
                        cmd_state['vx'] += V_STEP
                    elif is_key('s'):
                        cmd_state['vx'] -= V_STEP
                    elif is_key('q'):
                        cmd_state['vy'] += V_STEP
                    elif is_key('d'):
                        cmd_state['vy'] -= V_STEP
                    elif is_key('a'):
                        cmd_state['wz'] += W_STEP
                    elif is_key('e'):
                        cmd_state['wz'] -= W_STEP
                    elif is_key('r'):
                        cmd_state['z'] += Z_STEP
                    elif is_key('f'):
                        cmd_state['z'] -= Z_STEP
                    elif is_key('w'):
                        # reset to base on AZERTY use 'w' key to reset (maps to qwerty's 'z')
                        cmd_state['vx'] = float(base_vx)
                        cmd_state['vy'] = float(base_vy)
                        cmd_state['wz'] = float(base_wz)
                        cmd_state['z'] = float(base_z)

                # common non-layout keys handled after layout-specific keys
                if char == ' ':  # spacebar -> zero motion (toggle-ish)
                    cmd_state['vx'] = 0.0
                    cmd_state['vy'] = 0.0
                    cmd_state['wz'] = 0.0
                elif char == 'x':
                    # emergency stop: zero everything
                    cmd_state['vx'] = 0.0
                    cmd_state['vy'] = 0.0
                    cmd_state['wz'] = 0.0
                    cmd_state['z'] = float(base_z)
                else:
                    # special key handling (arrow keys)
                    try:
                        if key == keyboard.Key.up:
                            cmd_state['vx'] += V_STEP
                        elif key == keyboard.Key.down:
                            cmd_state['vx'] -= V_STEP
                        elif key == keyboard.Key.left:
                            # turn left
                            cmd_state['wz'] += W_STEP
                        elif key == keyboard.Key.right:
                            # turn right
                            cmd_state['wz'] -= W_STEP
                    except Exception:
                        # ignore if keyboard namespace not available
                        pass

        def _on_release(key):
            # stop listener on ESC
            try:
                if key == keyboard.Key.esc:
                    return False
            except Exception:
                pass

        # start listener in background
        _listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
        _listener.daemon = True
        _listener.start()
        if args_cli.keyboard_layout == "azerty":
            print('[INFO] Keyboard listener started (pynput). AZERTY layout: use Z/S/Q/D/A/E/R/F/Space/X/W to control commands.')
        else:
            print('[INFO] Keyboard listener started (pynput). QWERTY layout: use WASD/QE/RF/Space/X/Z to control commands.')
    except Exception:
        keyboard = None
        print('[WARN] pynput not available; keyboard control disabled. Install pynput to enable it.')
    # -------------------- end keyboard setup --------------------

    # helper to process a character or special key into cmd_state (shared by listeners)
    def process_key(char, special_key=None):
        """Update cmd_state based on char (single-letter) or special_key (pynput key)."""
        layout = args_cli.keyboard_layout if hasattr(args_cli, "keyboard_layout") else "qwerty"

        def is_key(k):
            return char == k if char is not None else False

        # layout-specific letter handling
        if layout == "qwerty":
            if is_key('w'):
                cmd_state['vx'] += V_STEP
            elif is_key('s'):
                cmd_state['vx'] -= V_STEP
            elif is_key('a'):
                cmd_state['vy'] += V_STEP
            elif is_key('d'):
                cmd_state['vy'] -= V_STEP
            elif is_key('q'):
                cmd_state['wz'] += W_STEP
            elif is_key('e'):
                cmd_state['wz'] -= W_STEP
            elif is_key('r'):
                cmd_state['z'] += Z_STEP
            elif is_key('f'):
                cmd_state['z'] -= Z_STEP
            elif is_key('z'):
                # reset to base
                cmd_state['vx'] = float(base_vx)
                cmd_state['vy'] = float(base_vy)
                cmd_state['wz'] = float(base_wz)
                cmd_state['z'] = float(base_z)
        else:  # azerty
            if is_key('z'):
                cmd_state['vx'] += V_STEP
            elif is_key('s'):
                cmd_state['vx'] -= V_STEP
            elif is_key('q'):
                cmd_state['vy'] += V_STEP
            elif is_key('d'):
                cmd_state['vy'] -= V_STEP
            elif is_key('a'):
                cmd_state['wz'] += W_STEP
            elif is_key('e'):
                cmd_state['wz'] -= W_STEP
            elif is_key('r'):
                cmd_state['z'] += Z_STEP
            elif is_key('f'):
                cmd_state['z'] -= Z_STEP
            elif is_key('w'):
                # reset to base on AZERTY use 'w' key to reset (maps to qwerty's 'z')
                cmd_state['vx'] = float(base_vx)
                cmd_state['vy'] = float(base_vy)
                cmd_state['wz'] = float(base_wz)
                cmd_state['z'] = float(base_z)

        # common keys
        if char == ' ':  # spacebar -> zero motion (toggle-ish)
            cmd_state['vx'] = 0.0
            cmd_state['vy'] = 0.0
            cmd_state['wz'] = 0.0
        elif char == 'x':
            # emergency stop: zero everything
            cmd_state['vx'] = 0.0
            cmd_state['vy'] = 0.0
            cmd_state['wz'] = 0.0
            cmd_state['z'] = float(base_z)

        # arrow / special key handling (when available via pynput)
        try:
            if special_key is not None:
                if special_key == keyboard.Key.up:
                    cmd_state['vx'] += V_STEP
                elif special_key == keyboard.Key.down:
                    cmd_state['vx'] -= V_STEP
                elif special_key == keyboard.Key.left:
                    cmd_state['wz'] += W_STEP
                elif special_key == keyboard.Key.right:
                    cmd_state['wz'] -= W_STEP
                elif special_key == keyboard.Key.esc:
                    return False
        except Exception:
            # keyboard namespace may not be available for stdin path
            pass

    # provide an stdin-based teleop (useful over RDP). This reads single chars from the
    # terminal without requiring Enter and updates cmd_state. It runs on a background thread.
    if args_cli.stdin_teleop:
        import threading

        def _stdin_thread():
            try:
                import sys, tty, termios

                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                print('[INFO] stdin teleop enabled; focus the terminal and press keys to control.')
                while True:
                    ch = sys.stdin.read(1)
                    if not ch:
                        continue
                    with cmd_lock:
                        # feed into the same processor (no special_key available)
                        if process_key(ch, None) is False:
                            break
            except Exception:
                print('[WARN] stdin teleop not available on this platform.')
                return
            finally:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass

        t = threading.Thread(target=_stdin_thread, daemon=True)
        t.start()

    # optionally disable resampler if present
    if args_cli.disable_resampler and hasattr(env.unwrapped, "next_resample_step"):
        env.unwrapped.next_resample_step[:] = 10 ** 12

    # initial observations
    obs = env.get_observations()
    timestep = 0

    # play loop
    print("[INFO] Starting playback loop. Press Ctrl-C to exit.")
    t_0 = time.time()
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # compute commanded values (optionally modulated)
            if args_cli.cmd_wave:
                t = time.time() - t_0
                vx = base_vx + 0.0 * math.sin(t / 5.0)
                vy = base_vy + 0.0 * math.sin(t / 7.0)
                wz = base_wz + 0.00 * math.sin(t / 3.0)
                zt = base_z + 0.1 * math.sin(t)
                # print(math.sin(t))
            else:
                # use current keyboard-updated command state (thread-safe)
                with cmd_lock:
                    vx = cmd_state.get('vx', float(base_vx))
                    vy = cmd_state.get('vy', float(base_vy))
                    wz = cmd_state.get('wz', float(base_wz))
                    zt = cmd_state.get('z', float(base_z))
                    # print(cmd_state.get('vx'))

            # assign to env tensors (broadcast to all envs)
            cmds = torch.tensor([vx, vy, wz], dtype=torch.float32, device=device).view(1, 3).expand(num_envs, -1)
            zcmds = torch.tensor([zt], dtype=torch.float32, device=device).view(1, 1).expand(num_envs, -1)
            try:
                env.unwrapped.vel_commands[:] = cmds
                env.unwrapped._z_command[:] = zcmds
            except Exception:
                # fallback: if wrapper layers prevent direct assignment, try per-subenv
                if hasattr(env, "envs"):
                    for sub in env.envs:
                        try:
                            sub.unwrapped.vel_commands[:] = cmds[:1]
                            sub.unwrapped._z_command[:] = zcmds[:1]
                        except Exception:
                            pass

            # get the (possibly normalized) observations after assignment
            obs = env.get_observations()
            # compute action
            actions = policy(obs)
            # print("Base height : ",obs["policy"][0,12])
            # step the env
            obs, _, _, _ = env.step(actions)

        # video termination logic
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator and env
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

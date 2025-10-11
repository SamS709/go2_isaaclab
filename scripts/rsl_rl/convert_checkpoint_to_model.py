"""Convert an RSL-RL training checkpoint into exported inference models.

This loads the checkpoint via the same runner code used in play.py, extracts the trained
policy network and normalizer, and exports a Torch JIT (`policy.pt`) and ONNX (`policy.onnx`) file
into an `exported/` directory next to the checkpoint.

The script prefers to reuse the isaaclab runner/exporter flow so the exported model
matches what's used by the play scripts. However, many isaaclab modules import
the Omniverse/IsaacSim packages (`omni.*`) at module-import time. If those packages
are not available in the active Python environment this script will gracefully fall
back to a "checkpoint inspection" mode that will attempt to extract any state_dicts
from the checkpoint and write them out so the user can re-run the conversion inside
an environment that has the full runtime.

Usage example:
    python convert_checkpoint_to_model.py --checkpoint /path/to/checkpoint.pt

If the checkpoint is inside a run directory produced by training, you can point to the
checkpoint file directly (the script will write exports in the same folder).
"""

import argparse
import os
import sys
import time

# local imports for CLI compatibility with other scripts (kept optional)
try:
    import cli_args  # isort: skip
except Exception:
    cli_args = None

import torch
import gymnasium as gym

# parse args
parser = argparse.ArgumentParser(description="Convert RSL-RL checkpoint to inference models (JIT + ONNX).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the RSL-RL checkpoint file.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--device", type=str, default=None, help="Device for runner creation (e.g. cuda:0).")
parser.add_argument("--export-onnx", action="store_true", help="Also export ONNX alongside JIT.")
parser.add_argument("--headless", action="store_true", help="Run without launching the Omniverse app (not required).")
args_cli, hydra_args = parser.parse_known_args()


def fallback_inspect_and_save(checkpoint_path: str, export_dir: str) -> None:
    """Fallback path when isaaclab/omni imports are unavailable.

    Attempts to torch.load the checkpoint and save any discovered state_dicts (policy,
    model, or generic state_dict) into the export directory so the user can re-run
    the conversion in a proper IsaacSim/Omniverse-enabled environment.
    """
    print("[WARN] Full isaaclab runtime not available. Falling back to checkpoint inspection mode.")
    os.makedirs(export_dir, exist_ok=True)
    print(f"[INFO] Loading checkpoint (cpu) for inspection: {checkpoint_path}")
    try:
        data = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        # As a last resort, copy the raw file to the export dir
        raw_dst = os.path.join(export_dir, "raw_checkpoint.pth")
        try:
            import shutil

            shutil.copy(checkpoint_path, raw_dst)
            print(f"[INFO] Copied raw checkpoint to: {raw_dst}")
        except Exception as e2:
            print(f"[ERROR] Failed to copy raw checkpoint: {e2}")
        return

    # If the checkpoint is a dict, look for common state_dict keys
    if isinstance(data, dict):
        saved_any = False
        candidates = [
            "policy_state_dict",
            "actor_critic_state_dict",
            "model_state_dict",
            "state_dict",
            "policy",
        ]
        for key in candidates:
            if key in data and isinstance(data[key], dict):
                dst = os.path.join(export_dir, f"{key}.pth")
                torch.save(data[key], dst)
                print(f"[INFO] Saved '{key}' to: {dst}")
                saved_any = True

        # If there is a nested 'alg' or 'runner' structure, try to inspect it
        for key in ("alg", "runner", "algorithm"):
            if key in data and isinstance(data[key], dict):
                inner = data[key]
                for subk, subv in inner.items():
                    if isinstance(subv, dict) and any(x in subv for x in ("state_dict", "policy_state_dict")):
                        dst = os.path.join(export_dir, f"{key}_{subk}.pth")
                        torch.save(subv, dst)
                        print(f"[INFO] Saved nested dict '{key}.{subk}' to: {dst}")
                        saved_any = True

        if not saved_any:
            # Nothing obvious found; save the entire dict for offline inspection
            dst = os.path.join(export_dir, "checkpoint_dict_dump.pth")
            torch.save(data, dst)
            print(f"[INFO] No explicit state_dict found. Saved full checkpoint dict to: {dst}")
    else:
        # The checkpoint might be a scripted/traced module or other object. Save it raw.
        dst = os.path.join(export_dir, "checkpoint_object.pth")
        try:
            torch.save(data, dst)
            print(f"[INFO] Saved checkpoint object to: {dst}")
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint object: {e}")
            # copy raw file as final fallback
            import shutil

            raw_dst = os.path.join(export_dir, "raw_checkpoint.pth")
            try:
                shutil.copy(checkpoint_path, raw_dst)
                print(f"[INFO] Copied raw checkpoint to: {raw_dst}")
            except Exception as e2:
                print(f"[ERROR] Failed to copy raw checkpoint: {e2}")


def main():
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    export_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")

    # Try to import the isaaclab runtime and run the normal conversion flow. If any
    # import fails (commonly due to missing omni/Omniverse packages), fall back to
    # the checkpoint-only inspection mode above.
    try:
        # Launch AppLauncher early so the omni.* packages are available to downstream imports.
        # Many isaaclab modules import omni.* during import time. We default device to 'cpu'
        # if the user did not provide one to avoid AppLauncher device resolution errors.
        if args_cli.device is None:
            args_cli.device = "cpu"

        # Import inside the try block to catch missing omniverse dependencies
        from isaaclab.app import AppLauncher
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner
        from isaaclab.envs import (
            DirectMARLEnv,
            DirectMARLEnvCfg,
            DirectRLEnvCfg,
            ManagerBasedRLEnvCfg,
            multi_agent_to_single_agent,
        )
        from isaaclab.utils.dict import print_dict
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

        # Create AppLauncher to make omni.* available to imports that expect it.
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        # We don't need to launch the SimulationApp to export the policy. Keep args parsing compatible
        # with other scripts, but skip AppLauncher event loop here.
        simulation_app = None

        # Create a minimal no-op gym env that satisfies the runner constructor.
        class _DummyEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.unwrapped = self
                self.observation_space = None
                self.action_space = None
                self.device = args_cli.device or "cpu"

            def get_observations(self):
                return {}

        env = _DummyEnv()

        # load a minimal fake agent cfg so runner can be constructed. We'll default to OnPolicyRunner.
        agent_cfg = RslRlBaseRunnerCfg()
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)

        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        runner.load(checkpoint_path)

        print("[INFO] Extracting trained policy for export")
        policy = runner.get_inference_policy(device=(args_cli.device or "cpu"))

        # extract the neural network module
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        # extract normalizer if present
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        print(f"[INFO] Exporting JIT to: {export_dir}")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")

        if args_cli.export_onnx:
            print(f"[INFO] Exporting ONNX to: {export_dir}")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")

        print("[INFO] Export completed.")

    except Exception as e:  # catch ImportError and other runtime issues
        print(f"[WARN] Full export flow failed or runtime not available: {e}")
        fallback_inspect_and_save(checkpoint_path, export_dir)


if __name__ == "__main__":
    main()

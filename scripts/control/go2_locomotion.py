# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates an interactive demo with the H1 rough terrain environment.

.. code-block:: bash

    # Usage
    python scripts/control/go2_locomotion.py --checkpoint pretrained_checkpoint/model_24600.pt
"""


import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import scripts.rsl_rl.cli_args as cli_args  # isort: skip


from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates an interactive demo with the Go2 robot environment."
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import torch

import carb
import omni
from isaacsim.core.utils.stage import get_current_stage
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY
from go2_isaaclab.tasks.direct.go2_isaaclab.go2_asymmetric_env_cfg import Go2AsymmetricEnvCfg

TASK = "Isaac-Velocity-Go2-Asymmetric-v0"
RL_LIBRARY = "rsl_rl"


class Go2FlatDemo:
    """This class provides an interactive demo for the GO2 flat terrain environment.
    It loads a pre-trained checkpoint for the Isaac-Velocity-Go2-Direct-v0, trained with RSL RL
    and defines a set of keyboard commands for directing motion of selected robots.

    A robot can be selected from the scene through a mouse click. Once selected, the following
    keyboard controls can be used to control the robot:

    * UP: go forward
    * LEFT: turn left
    * RIGHT: turn right
    * DOWN: stop
    * C: switch between third-person and perspective views
    * ESC: exit current third-person view"""

    def __init__(self):
        """Initializes environment config designed for the interactive model and sets up the environment,
        loads pre-trained checkpoints, and registers keyboard events."""
        agent_cfg: RslRlBaseRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        
        # Try to load checkpoint
        if args_cli.checkpoint:
            # Use user-specified checkpoint
            checkpoint = args_cli.checkpoint
            print(f"[INFO] Loading checkpoint from: {checkpoint}")
        else:
            # Try to get pretrained checkpoint
            checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
            if checkpoint is None:
                raise ValueError(
                    f"No pretrained checkpoint found for task '{TASK}'. "
                    f"Please train a model first or specify a checkpoint path using --checkpoint argument."
                )
            print(f"[INFO] Loading pretrained checkpoint: {checkpoint}")
        
        # create environment configuration
        env_cfg = Go2AsymmetricEnvCfg()
        env_cfg.scene.num_envs = 25
        env_cfg.episode_length_s = 1000000
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        env_cfg.commands.base_pos.ranges.z_pos = (0.3, 0.3)  # Fixed height at 0.3m
        
        # Create environment using gym.make (proper way to initialize)
        env = gym.make(TASK, cfg=env_cfg, render_mode=None)
        # wrap around environment for rsl-rl
        self.env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        self.device = self.env.unwrapped.device
        # load previously trained model
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
        # obtain the trained policy for inference
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 4, device=self.device)
        # Access command manager from DirectRLEnv (uses _commands attribute)
        self.commands[:, 0:3] = self.env.unwrapped._commands.get_command("base_velocity")
        self.commands[:, 3] = self.env.unwrapped._commands.get_command("base_pos")[:, 0]  # Squeeze to 1D
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)

    def create_camera(self):
        """Creates a camera to be used for third-person view."""
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def set_up_keyboard(self):
        """Sets up interface for keyboard input and registers the desired keys for control."""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        T = 1
        R = 0.5
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([-T, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([T, 0.0, -R], device=self.device),
            "RIGHT": torch.tensor([T, 0.0, R], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0, 0.3], device=self.device),
            "Z_UP":torch.tensor(0.35,device = self.device),
            "Z_DOWN":torch.tensor(0.25,device = self.device),
            
        }

    def _on_keyboard_event(self, event):
        MAX_VEL = 1.0
        MIN_HEIGHT = 0.2
        MAX_HEIGHT = 0.4
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Arrow keys map to pre-defined command vectors to control navigation of robot
            print(event.input.name)
            if event.input.name == "UP":
                if self.commands[self._selected_id, 0] <= MAX_VEL:
                    self.commands[self._selected_id,:3] += torch.tensor([0.1,0.0,0.0],device = self.device)
            elif event.input.name == "DOWN":
                if self.commands[self._selected_id, 0] >= -MAX_VEL:
                    self.commands[self._selected_id,:3] -= torch.tensor([0.1,0.0,0.0],device = self.device)
            elif event.input.name == "RIGHT":
                if self.commands[self._selected_id, 1] >= -MAX_VEL:
                    self.commands[self._selected_id,:3] -= torch.tensor([0.0,0.1,0.0],device = self.device)
            elif event.input.name == "LEFT":
                if self.commands[self._selected_id, 1] <= MAX_VEL:
                    self.commands[self._selected_id,:3] += torch.tensor([0.0,0.1,0.0],device = self.device)
            elif event.input.name == "F":
                if self.commands[self._selected_id, 1] <= MAX_VEL:
                    self.commands[self._selected_id,:3] += torch.tensor([0.0,0.0,0.1],device = self.device)
            elif event.input.name == "G":
                if self.commands[self._selected_id, 1] >= -MAX_VEL:
                    self.commands[self._selected_id,:3] -= torch.tensor([0.0,0.0,0.1],device = self.device)
            # Escape key exits out of the current selected robot view
            elif event.input.name == "E":
                if self.commands[self._selected_id,3] <= MAX_HEIGHT:
                    self.commands[self._selected_id,3] += torch.tensor(0.02, device = self.device)
            elif event.input.name == "R":
                if self.commands[self._selected_id,3] >= MIN_HEIGHT:
                    self.commands[self._selected_id,3] -= torch.tensor(0.02, device = self.device)
            elif event.input.name == "SPACE":
                self.commands[self._selected_id,:] = self._key_to_control["ZEROS"]
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            # C key swaps between third-person and perspective views
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
        # On key release, the robot stops moving
        

    def update_selected_object(self):
        """Determines which robot is currently selected and whether it is a valid H1 robot.
        For valid robots, we enter the third-person view for that robot.
        When a new robot is selected, we reset the command of the previously selected
        to continue random commands."""

        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            # a valid robot was selected, update the camera to go into third-person view
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a Go2 robot")

        # Reset commands for previously selected robot if a new one is selected
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped._commands.reset([self._previous_selected_id])
            self.commands[:, 0:3] = self.env.unwrapped._commands.get_command("base_velocity")
            self.commands[:, 3] = self.env.unwrapped._commands.get_command("base_pos")[:, 0]  # Squeeze to 1D

    def _update_camera(self):
        """Updates the per-frame transform of the third-person view camera to follow
        the selected robot's torso transform."""

        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]  # - env.scene.env_origins
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]

        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)


def main():
    """Main function."""
    demo_go2 = Go2FlatDemo()
    obs, _ = demo_go2.env.reset()
    while simulation_app.is_running():
        # check for selected robots
        demo_go2.update_selected_object()
        with torch.inference_mode():
            # Update the environment's actual commands (this will make debug vis work)
            demo_go2.env.unwrapped._commands._terms["base_velocity"].command[:] = demo_go2.commands[:, 0:3]
            demo_go2.env.unwrapped._commands._terms["base_pos"].command[:] = demo_go2.commands[:, 3:4]
            
            # Run policy
            print(obs["policy"].shape)
            action = demo_go2.policy(obs)
            obs, _, _, _ = demo_go2.env.step(action)
            
            # Update observation with current commands (for policy input)
            obs[:, 6:10] = demo_go2.commands


if __name__ == "__main__":
    main()
    simulation_app.close()

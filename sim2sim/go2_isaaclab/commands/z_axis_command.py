"""
Custom Z-Axis Position Command Term Implementation

This module implements a command term that samples z-axis position commands
for use with Isaac Lab's command manager system.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .z_axis_command_cfg import ZAxisCommandCfg


class ZAxisCommand(CommandTerm):
    """Command term that generates z-axis position commands.
    
    This command term samples z-axis target positions within a specified range
    and can be used for tasks like altitude control for drones or height control
    for jumping robots.
    """

    cfg: ZAxisCommandCfg
    """The configuration for the command term."""

    def __init__(self, cfg: ZAxisCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the z-axis command term.

        Args:
            cfg: The configuration for the command term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # create command buffer: [num_envs, 1] for z-position command
        self._command = torch.zeros(self.num_envs, 1, device=self.device)
        
        # create goal tracking buffer for visualization and metrics
        self._goal_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # get asset information (e.g., robot base)
        self._asset = env.scene[cfg.asset_name]
        
        # initialize metrics
        self.metrics["error_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_reached"] = torch.zeros(self.num_envs, device=self.device)

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """The z-axis command tensor. Shape is (num_envs, 1)."""
        return self._command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update the metrics based on the current state."""
        # get current z-position of the asset
        current_z_pos = self._asset.data.root_pos_w[:, 2]
        
        # compute z-axis error
        z_error = torch.abs(self._command[:, 0] - current_z_pos)
        self.metrics["error_z"] = z_error
        
        # check if goal is reached (within tolerance)
        self._goal_reached = z_error < self.cfg.goal_tolerance
        self.metrics["goal_reached"] = self._goal_reached.float()

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the z-axis position command for the specified environments.
        
        Args:
            env_ids: The environment indices to resample commands for.
        """
        # sample new z-axis position targets uniformly within the specified range
        self._command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.ranges.z_pos
        )

    def _update_command(self):
        """Update the command based on the current state.
        
        This can be used to modify the command during execution,
        e.g., for smooth transitions or adaptive behavior.
        """
        # For simple position commands, no update is needed
        # But you could implement things like:
        # - Smooth command transitions
        # - Velocity-based commands that change over time
        # - Adaptive commands based on robot state
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects."""
        if debug_vis:
            if not hasattr(self, "_goal_visualizer"):
                # create visualization markers for z-axis goals
                marker_cfg = self.cfg.goal_visualizer_cfg.replace(prim_path="/Visuals/Command/z_goal")
                self._goal_visualizer = VisualizationMarkers(marker_cfg)
            # set visibility
            self._goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "_goal_visualizer"):
                self._goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback for debug visualization."""
        if hasattr(self, "_goal_visualizer") and self._goal_visualizer.is_visible:
            # get current asset positions
            current_pos = self._asset.data.root_pos_w.clone()
            
            # set z-coordinate to the command target
            current_pos[:, 2] = self._command[:, 0]
            
            # update marker positions
            self._goal_visualizer.visualize(current_pos)



# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2 teacher-student distillation environment for Direct RL.

This implements the same teacher-student distillation concept as the G1 29DOF example,
but adapted for Direct RL environments instead of manager-based.
"""

from __future__ import annotations

import torch
from .go2_env import Go2Env, quat_apply_inverse
from .go2_asymmetric_env_cfg import Go2AsymmetricEnvCfg


class Go2AsymmetricEnv(Go2Env):
    """
    Go2 environment for teacher-student distillation.
    
    Key concepts:
    - Critic gets PRIVILEGED observations (full state including linear velocity)
    - Actor gets LIMITED observations (no linear velocity - must infer from other signals)
    - Both share the same action space
    """
    
    cfg: Go2AsymmetricEnvCfg

    def __init__(self, cfg: Go2AsymmetricEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _get_observations(self) -> dict:
        """
        Return both actor and critic observations.
        
        Returns:
            dict with keys:
                - "policy": actor observations (limited, 48 dims)
                - "critic": critic observations (privileged, 51 dims)
        """
        randomize = True
        self._previous_actions = self._actions.clone()
        
        # Update gait phase (increment by dt * frequency, wrap to [0, 1])
        self.phase = (self.phase + self.step_dt * self.gait_frequency) % 1.0
        
        # Encode phase as sin/cos (continuous representation)
        sin_phase = torch.sin(2 * torch.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * self.phase).unsqueeze(1)
        
        
        base_lin_vel_noisy = self._robot.data.root_lin_vel_b + (2.0 * torch.rand_like(self._robot.data.root_lin_vel_b) - 1.0) * float(0.1) * randomize
        base_ang_vel_noisy = self._robot.data.root_ang_vel_b + (2.0 * torch.rand_like(self._robot.data.root_ang_vel_b) - 1.0) * float(0.2) * randomize
        projected_gravity_noisy = quat_apply_inverse(
            self._robot.data.root_quat_w, torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        ) + (2.0 * torch.rand_like(self._robot.data.projected_gravity_b) - 1.0) * float(0.05)        
        joint_pos_noisy = (self._robot.data.joint_pos - self._robot.data.default_joint_pos) + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.01) * randomize
        joint_vel_noisy = self._robot.data.joint_vel + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.5) * randomize
        velocity_commands = self._commands.get_command("base_velocity")
        position_commands = self._commands.get_command("base_pos")
        
        # Get foot contact states (binary: 1 if in contact, 0 otherwise)
        foot_contacts = (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids], dim=-1) > 1.0).float()
        
        # Actor observations (NO linear velocity - this is the key limitation!)
        # Components: base_ang_vel(3) + proj_gravity(3) + vel_cmd(3) + pos_cmd(1) + joint_pos(12) + joint_vel(12) + actions(12) + sin_phase(1) + cos_phase(1) + contacts(4) = 52
        actor_obs = torch.cat(
            [
                base_ang_vel_noisy,                # 3
                projected_gravity_noisy,           # 3
                velocity_commands,                 # 3
                position_commands,                 # 1
                joint_pos_noisy,                   # 12
                joint_vel_noisy,                   # 12
                self._actions,                     # 12
                sin_phase,                         # 1
                cos_phase,                         # 1
                foot_contacts,                     # 4
            ],
            dim=-1,
        )
        # Critic observations (HAS linear velocity - privileged information!)
        # Components: base_lin_vel(3) + base_ang_vel(3) + proj_gravity(3) + vel_cmd(3) + pos_cmd(1) + joint_pos(12) + joint_vel(12) + actions(12) + sin_phase(1) + cos_phase(1) + contacts(4) = 55
        critic_obs = torch.cat(
            [
                base_lin_vel_noisy,                # 3 - PRIVILEGED!
                base_ang_vel_noisy,                # 3
                projected_gravity_noisy,           # 3
                velocity_commands,                 # 3
                position_commands,                 # 1
                joint_pos_noisy,                   # 12
                joint_vel_noisy,                   # 12
                self._actions,                     # 12
                sin_phase,                         # 1
                cos_phase,                         # 1
                foot_contacts,                     # 4
            ],
            dim=-1,
        )
        
        # Apply delay buffer if enabled
        if self.delay:
            actor_obs = self._buffer.compute(actor_obs)
            # Note: typically you'd want a separate buffer for teacher, or no delay for teacher
            # For simplicity, using same delay here
        
        return {
            "policy": actor_obs,    # Actor network sees this (47 dims)
            "critic": critic_obs,   # Critic network sees this (50 dims)
        }



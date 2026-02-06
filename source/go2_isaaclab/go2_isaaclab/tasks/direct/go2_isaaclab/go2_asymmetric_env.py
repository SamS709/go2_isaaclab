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
from .go2_env import Go2LidarEnv, Go2Env, quat_apply_inverse
from .go2_asymmetric_env_cfg import Go2AsymmetricEnvCfg, Go2AsymmetricLidarEnvCfg

torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)

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
        base_lin_vel_noisy = self._robot.data.root_lin_vel_b + (2.0 * torch.rand_like(self._robot.data.root_lin_vel_b) - 1.0) * float(0.01) * randomize
        base_ang_vel_noisy = self._robot.data.root_ang_vel_b + (2.0 * torch.rand_like(self._robot.data.root_ang_vel_b) - 1.0) * float(0.02) * randomize
        projected_gravity_noisy = quat_apply_inverse(
            self._robot.data.root_quat_w, torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        ) + (2.0 * torch.rand_like(self._robot.data.projected_gravity_b) - 1.0) * float(0.05)        
        joint_pos_noisy = (self._robot.data.joint_pos - self._robot.data.default_joint_pos) + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.01) * randomize
        joint_vel_noisy = self._robot.data.joint_vel + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.05) * randomize
        velocity_commands = self._commands.get_command("base_velocity")
        position_commands = self._commands.get_command("base_pos")
        foot_contacts = (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids], dim=-1) > 1.0).float()
        
        # Actor observations (NO linear velocity)
        actor_obs = torch.cat(
            [
                base_ang_vel_noisy,              
                projected_gravity_noisy,         
                velocity_commands,               
                position_commands,               
                joint_pos_noisy,                  
                joint_vel_noisy,                  
                self._actions,                    
                foot_contacts,                   
            ],
            dim=-1,
        )
        # Critic observations (HAS linear velocity)
        critic_obs = torch.cat(
            [
                base_lin_vel_noisy,                
                base_ang_vel_noisy,               
                projected_gravity_noisy,          
                velocity_commands,                
                position_commands,                
                joint_pos_noisy,                   
                joint_vel_noisy,                   
                self._actions,                            
                foot_contacts,                     
            ],
            dim=-1,
        )
        
        # Apply delay buffer if enabled
        if self.delay:
            actor_obs = self._buffer.compute(actor_obs)
        
        return {
            "policy": actor_obs,    # Actor network 
            "critic": critic_obs,   # Critic network 
        }

class Go2AsymmetricLidarEnv(Go2LidarEnv):
    cfg: Go2AsymmetricLidarEnvCfg

    def __init__(self, cfg: Go2AsymmetricLidarEnvCfg, render_mode: str | None = None, **kwargs):
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
        base_lin_vel_noisy = self._robot.data.root_lin_vel_b + (2.0 * torch.rand_like(self._robot.data.root_lin_vel_b) - 1.0) * float(0.01) * randomize
        base_ang_vel_noisy = self._robot.data.root_ang_vel_b + (2.0 * torch.rand_like(self._robot.data.root_ang_vel_b) - 1.0) * float(0.02) * randomize
        projected_gravity_noisy = quat_apply_inverse(
            self._robot.data.root_quat_w, torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        ) + (2.0 * torch.rand_like(self._robot.data.projected_gravity_b) - 1.0) * float(0.05)        
        joint_pos_noisy = (self._robot.data.joint_pos - self._robot.data.default_joint_pos) + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.01) * randomize
        joint_vel_noisy = self._robot.data.joint_vel + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.05) * randomize
        velocity_commands = self._commands.get_command("base_velocity")
        position_commands = self._commands.get_command("base_pos")
        
        # Get foot contact states (binary: 1 if in contact, 0 otherwise)
        foot_contacts = (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids], dim=-1) > 1.0).float()
        # print("Feet names:", [self._robot.body_names[i] for i in self._feet_ids])        
        
        height_map = self._get_lidar_obs()
        height_map_noisy = height_map + (2.0 * torch.rand_like(height_map) - 1.0) * float(0.1)
        # print(height_map_noisy)
        height_map_flat = height_map.reshape(self.num_envs, -1)
        height_map_flat_noisy = height_map_flat + (2.0 * torch.rand_like(height_map_flat) - 1.0) * float(0.1)
        # print(height_map_flat_noisy)
        actor_obs = torch.cat(
            [
                base_ang_vel_noisy,                # 3
                projected_gravity_noisy,           # 3
                velocity_commands,                 # 3
                position_commands,                 # 1
                joint_pos_noisy,                   # 12
                joint_vel_noisy,                   # 12
                self._actions,                     # 12
                foot_contacts,                     # 4
                height_map_flat_noisy,
            ],
            dim=-1,
        )
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
                foot_contacts,                     # 4
                height_map_flat_noisy,
            ],
            dim=-1,
        )
        
        if self.delay:
            actor_obs = self._buffer.compute(actor_obs)

        return {
            "policy": actor_obs,    
            "critic": critic_obs,   
        }



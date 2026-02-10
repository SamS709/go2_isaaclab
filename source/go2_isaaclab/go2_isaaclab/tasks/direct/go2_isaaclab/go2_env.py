# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from isaaclab.utils.buffers import DelayBuffer

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import CommandManager, CurriculumManager
from isaaclab.utils.math import quat_apply_inverse,  quat_apply, quat_conjugate

from .go2_env_cfg import Go2FlatEnvCfg, Go2LidarEnvCfg


class Go2Env(DirectRLEnv):
    cfg: Go2FlatEnvCfg

    def __init__(self, cfg: Go2FlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                # "lin_vel_dir",
                "track_ang_vel_z_exp",
                "track_base_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "joint_vel_l2",                
                "joint_torques_l2",
                "joint_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "flat_orientation_l2",
                "default_pos",
                "feet_var",
                "energy",
                "termination_penalty",
                "undesired_contacts",
                "dof_pos_limits"
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self._hip_ids, _ = self._contact_sensor.find_bodies(".*hip")
        self._calf_ids, _ = self._contact_sensor.find_bodies(".*calf")
                
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._commands = CommandManager(self.cfg.commands, self)
        
        self._lidar = None
        self.history_length = 2
        self.delay = True
        if self.delay == True:
            self._buffer = DelayBuffer(history_length=self.history_length, batch_size=self.num_envs, device=self.device)
            if self.history_length > 0:
                self._buffer.set_time_lag(
                    torch.randint(low=0, high=self.history_length, size=(self.num_envs,), device=self.device)
                )
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Update command manager (handles resampling and command updates)
        self._commands.compute(dt=self.step_dt)
        
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_lidar_obs(self):
        """Generate per-environment heightmaps from lidar data (fully vectorized).
        
        Returns:
            torch.Tensor: Heightmaps with shape [num_envs, x_cells, y_cells]
        """
        x_range = (-self.cfg.height_map_dist, self.cfg.height_map_dist)
        y_range = (-self.cfg.height_map_dist / 2.0, self.cfg.height_map_dist / 2.0)
        
        # Grid dimensions
        x_cells = int((x_range[1] - x_range[0]) * self.cfg.res)
        y_cells = int((y_range[1] - y_range[0]) * self.cfg.res)
        cells_per_env = x_cells * y_cells
        
        # Calculate rays: [num_envs, num_rays, 3]
        rays = self._lidar.data.pos_w.unsqueeze(1) - self._lidar.data.ray_hits_w
        
        # Mark valid rays before replacing inf
        ray_hit = torch.isfinite(rays).all(dim=-1)
       
       
        offset_robot_frame = torch.tensor(self.cfg.lidar_offset, device=rays.device).unsqueeze(0).repeat(rays.shape[0], 1)  # [num_envs, 3]
        offset_world_frame = quat_apply(self._robot.data.root_quat_w, offset_robot_frame)  # [num_envs, 3]
        rays += offset_world_frame.unsqueeze(1)
        # Rotate from world to robot frame
        # Reshape rays to [num_envs * num_rays, 3] for quat_apply
        num_envs, num_rays, _ = rays.shape
        rays_flat = rays.reshape(num_envs * num_rays, 3)
        # Expand quaternions to match: [num_envs, 4] -> [num_envs * num_rays, 4]
        quat_expanded = quat_conjugate(self._robot.data.root_quat_w).unsqueeze(1).expand(num_envs, num_rays, 4).reshape(num_envs * num_rays, 4)
        rays_flat = quat_apply(quat_expanded, rays_flat)
        rays = rays_flat.reshape(num_envs, num_rays, 3)
        
        # Convert to grid indices: [num_envs, num_rays]
        x_idx = ((rays[:, :, 0] - x_range[0]) * self.cfg.res).long()
        y_idx = ((rays[:, :, 1] - y_range[0]) * self.cfg.res).long()
        
        # Validate and flatten
        valid = (
            ray_hit &
            (x_idx >= 0) & (x_idx < x_cells) &
            (y_idx >= 0) & (y_idx < y_cells)
        )
        
        # Create global indices: env_id * cells_per_env + x_idx * y_cells + y_idx
        env_ids = torch.arange(self.num_envs, device=rays.device).view(-1, 1).expand_as(x_idx)
        global_idx = (env_ids * cells_per_env + x_idx * y_cells + y_idx)[valid]
        z_vals = rays[:, :, 2][valid]
        
        # Single scatter operation for all environments
        height_map = torch.zeros(self.num_envs * cells_per_env, device=rays.device)
        if len(global_idx) > 0:
            height_map.scatter_reduce_(0, global_idx, z_vals, reduce='amax', include_self=False)
        
        # Reshape and set center region to 0 for each environment
        height_map = height_map.view(self.num_envs, x_cells, y_cells)
        height_map[height_map<0.0] = 0.0
        height_map[:, -1, :] = 0.0
        
        return height_map
    
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        self.projected_gravity = quat_apply_inverse(
            self._robot.data.root_quat_w, torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        )

        foot_contacts = (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids], dim=-1) > 1.0).float()
        
        # observations with noise added
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b + (2.0 * torch.rand_like(self._robot.data.root_lin_vel_b) - 1.0) * float(0.1),
                    self._robot.data.root_ang_vel_b + (2.0 * torch.rand_like(self._robot.data.root_ang_vel_b) - 1.0) * float(0.2),
                    self.projected_gravity + (2.0 * torch.rand_like(self.projected_gravity) - 1.0) * float(0.05),
                    self._commands.get_command("base_velocity"),
                    self._commands.get_command("base_pos"),
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.01),
                    self._robot.data.joint_vel + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.5),
                    self._actions,
                    foot_contacts,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        if self.delay == True:
            obs = self._buffer.compute(obs)
        observations = {"policy": obs}
        return observations
    
    def compute_tri(self, height_map):
        """
        Terrain Ruggedness Index: mean absolute height difference from neighbors
        Args:
            height_map: [num_envs, x_cells, y_cells]
        Returns:
            tri: [num_envs] - scalar roughness per environment
        """
        # Compute differences with all 8 neighbors using convolution
        kernel = torch.tensor([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=height_map.dtype, device=height_map.device) / 8.0
        
        # Add channel dimension for conv2d: [num_envs, 1, x, y]
        height_map_4d = height_map[:,:-1,:].unsqueeze(1)  # Don't slice yet!
        
        # Compute local mean of neighbors
        neighbor_mean = torch.nn.functional.conv2d(
            height_map_4d, 
            kernel.view(1, 1, 3, 3),
            padding=0
        ).squeeze(1)
        # Now compute differences and slice both together
        abs_diff = torch.abs(height_map[:,1:-2,1:-1] - neighbor_mean)  # Slice after subtraction
        # Get top n differences per environment
        n_sampled = 3
        abs_flat = abs_diff.flatten(start_dim=1)
        values, _ = torch.topk(abs_flat, n_sampled, dim=-1)
        tri = torch.mean(values, dim=-1)
        return tri

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        
        roughness_scale = 1.0
        if self._lidar is not None:
            height_map = self._get_lidar_obs()
            terrain_roughness = self.compute_tri(height_map)
            roughness_scale = torch.exp(-20.0 * terrain_roughness)

        lin_vel_error = torch.sum(torch.square(self._commands.get_command("base_velocity")[:, :2] - self._robot.data.root_lin_vel_b[:, :2]),dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25) 
        
       
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands.get_command("base_velocity")[:, 2] - self._robot.data.root_ang_vel_b[:, 2]) * roughness_scale
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25) #* roughness_scale
        # base_z_tracking
        if self._lidar is not None:
            base_z_error = torch.exp(-torch.square((self._commands.get_command("base_pos")[:,0] - self._robot.data.root_pos_w[:,2]))/0.0025)
            #base_z_error *= roughness_scale
        else:
            base_z_error = 0.0
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2]) * roughness_scale
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1) * roughness_scale
        # joint vel
        joint_vel = torch.sum(torch.square(torch.square(self._robot.data.joint_vel)), dim=1) * roughness_scale
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1) * roughness_scale
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1) * roughness_scale
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1) * roughness_scale
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum(torch.exp(-torch.square(0.5-last_air_time)/0.01) * first_contact, dim=1) * (
            torch.norm(self._commands.get_command("base_velocity")[:, :2], dim=1) > 0.1
        ) 
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1) * roughness_scale
        
        # stay around default pos:        
        cmd = torch.linalg.norm(self._commands.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        rew = torch.linalg.norm((self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1)
        def_pos = torch.where(torch.logical_or(cmd > 0.0, body_vel > self.cfg.velocity_threshold), rew, self.cfg.stand_still_scale * rew)  * roughness_scale
        
        #Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        last_contact_time = self._contact_sensor.data.last_contact_time[:, self._feet_ids]
        feet_var =  torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(torch.clip(last_contact_time, max=0.5), dim=1) * roughness_scale
        
        # energy consumption (power = |torque| * |velocity|)
        energy = torch.sum(torch.abs(self._robot.data.joint_vel) * torch.abs(self._robot.data.applied_torque), dim=-1) * roughness_scale
        
        # termination penalty (applied when base contacts ground)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1).float()
        # undesired contacts (penalize thigh contacts)
        u_contact = torch.max(torch.norm(net_contact_forces[:, :, self._thigh_ids + self._hip_ids + self._calf_ids], dim=-1), dim=1)[0] > 1.0
        undesired_contacts = torch.sum(u_contact, dim=1)
        
        # joint position limits (penalize exceeding soft limits)
        out_of_limits = -(
            self._robot.data.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 0]
        ).clip(max=0.0)
        out_of_limits += (
            self._robot.data.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 1]
        ).clip(min=0.0)
        dof_pos_limits = torch.sum(out_of_limits, dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            # "lin_vel_dir": lin_vel_dir * self.cfg.lin_vel_dir_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "track_base_z_exp": base_z_error * self.cfg.base_z_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "joint_vel_l2": joint_vel * self.cfg.joint_vel_reward_scale * self.step_dt,
            "joint_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "joint_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "default_pos": def_pos * self.cfg.respect_def_pos_reward_scale * self.step_dt,
            "feet_var": feet_var * self.cfg.feet_var_reward_scale * self.step_dt,
            "energy": energy * self.cfg.energy_reward_scale * self.step_dt,
            "termination_penalty": base_contact * self.cfg.termination_penalty_scale,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contacts_scale * self.step_dt,
            "dof_pos_limits": dof_pos_limits * self.cfg.dof_pos_limits_scale * self.step_dt
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        # Reset command manager (important for resampling commands)
        self._commands.reset(env_ids)
        
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        if self.delay == True:
            self._buffer.reset(env_ids.tolist())
            self._buffer.set_time_lag(
                    torch.randint(low=0, high=self.history_length, size=(self.num_envs,), device=self.device)
                )
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)



class Go2LidarEnv(Go2Env):
    cfg: Go2LidarEnvCfg

    def __init__(self, cfg: Go2LidarEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        
        
    def _setup_scene(self):
        super()._setup_scene()
        self._lidar = RayCaster(self.cfg.lidar_cfg)
        self.scene.sensors["lidar"] = self._lidar
        
        # Initialize curriculum manager for terrain difficulty progression
        self._curriculum = CurriculumManager(self.cfg.curriculum, self)
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Call parent reset
        super()._reset_idx(env_ids)
        
        # Update curriculum (adjusts terrain difficulty based on performance)
        self._curriculum.compute(env_ids)
   
    
        
    
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        self.projected_gravity = quat_apply_inverse(
            self._robot.data.root_quat_w, torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        )

        foot_contacts = (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids], dim=-1) > 1.0).float()
        
        # Get per-environment heightmaps: [num_envs, x_cells, y_cells]
        height_map = self._get_lidar_obs()
        
        
        # Flatten heightmap for each environment: [num_envs, x_cells * y_cells]
        height_map_flat = height_map.reshape(self.num_envs, -1)
        
        
        # observations with noise added
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b + (2.0 * torch.rand_like(self._robot.data.root_lin_vel_b) - 1.0) * float(0.1),
                    self._robot.data.root_ang_vel_b + (2.0 * torch.rand_like(self._robot.data.root_ang_vel_b) - 1.0) * float(0.2),
                    self.projected_gravity + (2.0 * torch.rand_like(self.projected_gravity) - 1.0) * float(0.05),
                    self._commands.get_command("base_velocity"),
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.01),
                    self._robot.data.joint_vel + (2.0 * torch.rand_like(self._robot.data.default_joint_pos) - 1.0) * float(0.5),
                    self._actions,
                    foot_contacts,
                    height_map_flat + (2.0 * torch.rand_like(height_map_flat) - 1.0) * float(0.01),
                )
                if tensor is not None
            ],
            dim=-1,
        )
        if self.delay == True:
            obs = self._buffer.compute(obs)
        observations = {"policy": obs}
        return observations

    
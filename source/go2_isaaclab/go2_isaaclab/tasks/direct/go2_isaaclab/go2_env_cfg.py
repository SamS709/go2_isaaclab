# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# The goal is to move forward

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
# Uncomment the two lines below for MuJoCo env in Newton Isaaclab's branch
# from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
# from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Import custom commands
from .commands.z_axis_command_cfg import ZAxisCommandCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=None
        ),
    )
    
    base_pos = ZAxisCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 6.0),
        debug_vis=True,
        ranges=ZAxisCommandCfg.Ranges(
            z_pos=(0.2, 0.4),
        ),
        goal_tolerance=0.05,
    )

@configclass
class EventCfg:
    """Configuration for randomization."""
    
# Comment reset_robot_joints if you are in Newton Isaaclab's branch)
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),  # Randomize joint positions by ±0.3 radians
            "velocity_range": (-0.05, 0.05),  # Small random initial velocities
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
    
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset", 
        params={
            "pose_range": {
                "x": (-1.0, 1.0),      # Random X position ±1m
                "y": (-1.0, 1.0),      # Random Y position ±1m  
                "z": (0.0, 0.1),       # Slight Z variation
                "roll": (-0.1, 0.1),   # Small roll variation (radians)
                "pitch": (-0.1, 0.1),  # Small pitch variation
                "yaw": (-3.14, 3.14),  # Full yaw randomization
            },
            "velocity_range": {
                "x": (-0.5, 0.5),      # Random initial linear velocity
                "y": (-0.5, 0.5),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),   # Random initial angular velocity
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    robot_joint_stiffness_and_damping = EventTerm(
      func=mdp.randomize_actuator_gains,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
          "stiffness_distribution_params": (0.9, 1.1),
          "damping_distribution_params": (0.9,1.1),
          "operation": "scale",
          "distribution": "uniform",
      },
  )
    
    
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.5, 0.7),
            "restitution_range": (0.0, 0.01),
            "num_buckets": 64,
        },
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.1, 0.1),
        },
    )


    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 7.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class Go2FlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    observation_space = 51
    state_space = 0

# classic imulation (comment if you are in Newton Isaaclab's branch)
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
    
# MuJoCo simulation (uncomment if you are in Newton Isaaclab's branch)
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
    # solver_cfg = MJWarpSolverCfg(
    #     njmax=150,
    #     ncon_per_env=35,
    #     ls_iterations=10,
    #     cone="pyramidal",
    #     ls_parallel=True,
    #     impratio=1,
    #     integrator="implicit",
    # )
    
    # newton_cfg = NewtonCfg(
    #     solver_cfg=solver_cfg,
    #     num_substeps=1,
    #     debug_mode=False,
    # )

    # sim: SimulationCfg = SimulationCfg(
    #     dt=1 / 200,
    #     render_interval=decimation,
    #     newton_cfg=newton_cfg,
    # )
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()
    
    commands: CommandsCfg = CommandsCfg()
    
    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales
    lin_vel_reward_scale = 1.5 # replace by 1.5 for env without damping and switfness randomization
    yaw_rate_reward_scale = 0.75
    base_z_reward_scale = 0.7
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -0.0002
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.25
    flat_orientation_reward_scale = -2.5
    feet_distance_reward_scale = 0.0



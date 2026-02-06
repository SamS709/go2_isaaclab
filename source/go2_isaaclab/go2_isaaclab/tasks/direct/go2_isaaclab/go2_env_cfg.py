# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# The goal is to move forward
import torch
from collections.abc import Sequence
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


# Import custom commands
from .commands.z_axis_command_cfg import ZAxisCommandCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

DEBUG_VIS = True

def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    # Use direct attributes from Go2Env instead of scene manager
    asset: Articulation = env._robot
    terrain: TerrainImporter = env._terrain
    command = env._commands.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=DEBUG_VIS,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=None
        ),
    )
    
    base_pos = ZAxisCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        debug_vis=DEBUG_VIS,
        ranges=ZAxisCommandCfg.Ranges(
            z_pos=(0.2, 0.4),
        ),
        goal_tolerance=0.05,
    )

@configclass
class EventCfg:
    """Configuration for randomization."""
    

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),  # Randomize joint positions by ±0.3 radians
            "velocity_range": (-0.05, 0.05),  # Small random initial velocities
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

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
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="startup",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (-1.0, 1.0),
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
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class Go2FlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    observation_space = 53
    state_space = 0
    visualize = False

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=168635 * 2,  # Increased from default to prevent buffer overflow
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
  
    # This terrain adds a little bit of noise so that the robot can walk on carpet or objects on the ground
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.0, 1.0),
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.6),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.4, noise_range=(0.00, 0.005), noise_step=0.008, border_width=0.25
                ),
            },
        ),
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=DEBUG_VIS,
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
    lin_vel_reward_scale = 1.0 # replace by 1.5 for env without damping and switfness randomization
    lin_vel_dir_scale = 0.5  # Reward for matching velocity direction
    yaw_rate_reward_scale = 0.75
    base_z_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_vel_reward_scale = -0.001
    joint_torque_reward_scale = -0.0002
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.1
    feet_air_time_reward_scale = 0.1
    flat_orientation_reward_scale = -2.5
    feet_distance_reward_scale = 0.0
    respect_def_pos_reward_scale = -0.07
    stand_still_scale = 5.0
    feet_var_reward_scale = -1.0
    energy_reward_scale = -2e-5
    termination_penalty_scale = -200.0  # Large penalty for falling/base contact
    undesired_contacts_scale = -1.0  # Penalty for thigh contacts
    dof_pos_limits_scale = -10.0  # Penalty for joints exceeding soft limits
    
    velocity_threshold = 0.3
    
    
    def __post_init__(self):
        """Post initialization to set debug_vis based on visualize flag."""
        # Update debug_vis for commands based on visualize attribute
        self.commands.base_velocity.debug_vis = self.visualize
        self.commands.base_pos.debug_vis = self.visualize


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=terrain_levels_vel)


@configclass
class Go2LidarEnvCfg(Go2FlatEnvCfg):
    
    
    ROUGH_TERRAINS_CFG.num_cols = 1
    ROUGH_TERRAINS_CFG.num_rows = 1
    curriculum: CurriculumCfg = CurriculumCfg()
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=DEBUG_VIS,
    )
    
    # Heightmap configuration
    height_map_dist = 1.0
    res = 6  # resolution of the heightmap (cells per meter)
    height_map_cells = int(2 * height_map_dist * res) ** 2  
    observation_space = 53 + height_map_cells  
    
    lidar_range = height_map_dist * 3.0 # * 1.4142135623730951  # sqrt(2)
    lidar_offset = (0.28945, 0.0, -0.04682)
    # Pre-computed quaternion (w, x, y, z) from euler angles (-pi, pi - 2.8782, -pi)
    lidar_rotation = (1.3132e-01, 3.7593e-08, 9.9134e-01, 3.7593e-08)
    
    lidar_cfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(
            pos=lidar_offset,
            rot=lidar_rotation,
        ),
        mesh_prim_paths=["/World/ground"],
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=128, vertical_fov_range=[-90.0, 90.0], horizontal_fov_range=[-180, 180], horizontal_res=2.0
        ),
        max_distance=lidar_range,
        debug_vis=DEBUG_VIS,
    )
    
    



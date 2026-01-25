# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example on using the raycaster sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

# Set torch print options
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip
import isaaclab.terrains as terrain_gen

from isaaclab.terrains.config import TerrainGeneratorCfg


##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG 

MAX_RAY_DIST = 1.0 # the rays make a circle of radius = sqrt(2) * MAX_RAY_DIST, so that the height map is fully 
OFFSET = (0.28945, 0.0, -0.04682) # offset for the lidar from the base
W, X, Y, Z = quat_from_euler_xyz(torch.tensor(-torch.pi), torch.tensor(torch.pi - 2.8782), torch.tensor(-torch.pi))
@configclass
class RaycasterSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # rough terrain with boxes
    ROUGH_TERRAINS_CFG.num_cols = 1
    ROUGH_TERRAINS_CFG.num_rows = 2
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
    },
),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # specs of the real lidar:
    '''
    360 * 90 degrees FOV
    translated by approx 0.3 m on x axis and -0.047 along z axis
    rotated by approx. (rpy) [-180, 15.1, -180]. (15.1 degrees along yaw !!!) 
    The values have been chosen debugging the vis 
    '''
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(
            pos=OFFSET,
            rot=(W, X, Y, Z),  
        ),
        mesh_prim_paths=["/World"],
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=64, vertical_fov_range=[-90.0, 90.0], horizontal_fov_range=[-180, 180], horizontal_res=1.0
        ),
        max_distance= MAX_RAY_DIST * 2.0 ** (1.0 / 2.0) ,
        debug_vis=not args_cli.headless,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    triggered = True
    countdown = 42

    # Simulate physics
    while simulation_app.is_running():

        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            # Offset robot 5 meters to the right (positive y-direction)
            root_state[:, 1] -= 4.25
            root_state[:, 0] += 0.7
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        
        targets = scene["robot"].data.default_joint_pos 
        targets += torch.rand_like(joint_pos) * 0.1
        # print("pos: ", scene["robot"].data.default_joint_pos.clone())        
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        print("-------------------------------")
        print("Robot base height (z):", scene["robot"].data.root_pos_w[:, 2])
        # print(scene["ray_caster"])
        rays = scene["ray_caster"].data.pos_w.unsqueeze(1) - scene["ray_caster"].data.ray_hits_w         
        rays[torch.isinf(rays)] = 0
        # Remove sensor offset from all axes
        rays[:, :, 0] += OFFSET[0]  # x offset
        rays[:, :, 1] += OFFSET[1] # y offset (no change)
        rays[:, :, 2] += OFFSET[2] # z offset
        print("Ray cast hit results: ", rays.shape)
        print((rays[:,:,0]))
        finite_x = rays[:,:,0][torch.isfinite(rays[:,:,0])]
        finite_y = rays[:,:,1][torch.isfinite(rays[:,:,1])]
        print("max x: ", torch.max(finite_x) if finite_x.numel() > 0 else "No finite values")
        print("min x: ", torch.min(finite_x) if finite_x.numel() > 0 else "No finite values")
        # print("mean y: ",torch.mean(rays[:,:,1]))        
        print("max y: ", torch.max(finite_y) if finite_y.numel() > 0 else "No finite values")
        print("min y: ", torch.min(finite_y) if finite_y.numel() > 0 else "No finite values")
        
        # Create height map using loop
        # res = 6  # resolution (cells per meter)
        # x_cells = 2 * int(MAX_RAY_DIST) * res
        # y_cells = 2 * int(MAX_RAY_DIST) * res
        # height_map = torch.full((x_cells, y_cells), float('-inf'), device=rays.device)
        
        # for i in range(x_cells):
        #     for j in range(y_cells):
        #         # Define grid cell boundaries
        #         x_min = -MAX_RAY_DIST + i / res
        #         x_max = -MAX_RAY_DIST + (i + 1) / res
        #         y_min = -MAX_RAY_DIST + j / res
        #         y_max = -MAX_RAY_DIST + (j + 1) / res
                
        #         # Filter rays in this cell
        #         mask = (rays[:, :, 0] >= x_min) & (rays[:, :, 0] < x_max) & \
        #                (rays[:, :, 1] >= y_min) & (rays[:, :, 1] < y_max) & \
        #                torch.isfinite(rays[:, :, 2])
                
        #         if mask.any():
        #             height_map[i, j] = torch.max(rays[:, :, 2][mask])
        
        # # Replace -inf with 0 for empty cells
        # height_map[height_map == float('-inf')] = 0.0
        
        # print(f"Height map shape: {height_map.shape}")
        # print(f"Height map min: {height_map.min()}, max: {height_map.max()}")
        # print(height_map)
        
        # Create height map from raycaster data
        res = 6  # resolution (cells per meter)
        x_range = (-MAX_RAY_DIST, MAX_RAY_DIST)
        y_range = (-MAX_RAY_DIST, MAX_RAY_DIST)
        
        # Create grid dimensions
        x_cells = int((x_range[1] - x_range[0]) * res)
        y_cells = int((y_range[1] - y_range[0]) * res)
        
        # Initialize height map with very high values for min operation
        height_map = torch.full((x_cells, y_cells), float('inf'), device=rays.device)
        
        # Flatten rays for easier processing: [num_envs * num_rays, 3]
        rays_flat = rays.reshape(-1, 3)
        
        # Filter out invalid rays (inf values)
        valid_mask = torch.isfinite(rays_flat).all(dim=1)
        rays_valid = rays_flat[valid_mask]
        
        # Convert x, y positions to grid indices
        x_idx = ((rays_valid[:, 0] - x_range[0]) * res).long()
        y_idx = ((rays_valid[:, 1] - y_range[0]) * res).long()
        z_vals = rays_valid[:, 2]
        
        # Filter indices that are within bounds
        valid_idx_mask = (x_idx >= 0) & (x_idx < x_cells) & (y_idx >= 0) & (y_idx < y_cells)
        x_idx = x_idx[valid_idx_mask]
        y_idx = y_idx[valid_idx_mask]
        z_vals = z_vals[valid_idx_mask]
        
        # Use scatter_reduce to get min z for each grid cell
        # Flatten 2D indices to 1D
        flat_indices = x_idx * y_cells + y_idx
        height_map_flat = height_map.flatten()
        
        # Get min z for each unique grid cell
        height_map_flat = height_map_flat.scatter_reduce(
            0, flat_indices, z_vals, reduce='amin', include_self=True
        )
        height_map = height_map_flat.reshape(x_cells, y_cells)
        
        # Replace inf with 0 or any default value for empty cells
        height_map[height_map == float('inf')] = 0.0
        
        print(f"Height map shape: {height_map.shape}")
        print(f"Height map min: {height_map.min()}, max: {height_map.max()}")
        print(height_map)
        
        if not triggered:
            if countdown > 0:
                countdown -= 1
                continue
            data = scene["ray_caster"].data.ray_hits_w.cpu().numpy()
            np.save("cast_data.npy", data)
            triggered = True
        else:
            continue


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = RaycasterSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

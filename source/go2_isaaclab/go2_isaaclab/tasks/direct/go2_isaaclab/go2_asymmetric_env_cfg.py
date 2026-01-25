# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for Go2 student environment (inference/play mode).
"""

from isaaclab.utils import configclass

from .go2_env_cfg import Go2FlatEnvCfg, Go2LidarEnvCfg


@configclass
class Go2AsymmetricEnvCfg(Go2FlatEnvCfg):
    """
    Configuration for student policy inference.
    
    This environment configuration is for playing trained distilled policies.
    It uses the same observation space as the student during distillation training.
    """

    observation_space = 50
    critic_observation_space = 53
    
    def __post_init__(self):
        super().__post_init__()
        # Typically use fewer environments for inference/evaluation
        self.scene.num_envs = 64
        
@configclass
class Go2AsymmetricLidarEnvCfg(Go2LidarEnvCfg):
    """
    Configuration for student policy inference.
    
    This environment configuration is for playing trained distilled policies.
    It uses the same observation space as the student during distillation training.
    """

    def __post_init__(self):
        super().__post_init__()
        # Typically use fewer environments for inference/evaluation
        self.scene.num_envs = 64
        # Calculate heightmap size as integer: (2 * height_map_dist * res)^2
        height_map_cells = int(2 * self.height_map_dist * self.res) ** 2
        self.observation_space = 50 + height_map_cells
        self.critic_observation_space = 50 + height_map_cells

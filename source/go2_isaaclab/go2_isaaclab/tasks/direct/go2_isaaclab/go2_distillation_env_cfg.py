# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for Go2 teacher-student distillation using Direct RL.

This demonstrates how to adapt manager-based distillation (like G1 29DOF example)
to work with Direct RL environments.
"""

from isaaclab.utils import configclass

from .go2_env_cfg import Go2FlatEnvCfg


@configclass
class Go2TeacherStudentEnvCfg(Go2FlatEnvCfg):
    """
    Configuration for teacher-student distillation.
    
    Key differences from standard training:
    - observation_space defines STUDENT observation size (reduced)
    - teacher_observation_space defines TEACHER observation size (privileged/full)
    - The environment must return dict with both "policy" (student) and "teacher" keys
    """
    
    # Student observations (limited - no linear velocity)
    # Components: base_ang_vel(3) + proj_gravity(3) + commands(3) + joint_pos(12) + joint_vel(12) + actions(12) = 45
    observation_space = 47
    
    # Teacher observations (privileged - has linear velocity)
    # Components: base_lin_vel(3) + base_ang_vel(3) + proj_gravity(3) + commands(3) + joint_pos(12) + joint_vel(12) + actions(12) = 48
    teacher_observation_space = 50
    
    def __post_init__(self):
        super().__post_init__()
        # Reduce number of environments for distillation training
        self.scene.num_envs = 256


@configclass
class Go2StudentFineTuneEnvCfg(Go2FlatEnvCfg):
    """
    Configuration for teacher-student distillation.
    
    Key differences from standard training:
    - observation_space defines STUDENT observation size (reduced)
    - teacher_observation_space defines TEACHER observation size (privileged/full)
    - The environment must return dict with both "policy" (student) and "teacher" keys
    """
    
    # Student observations (limited - no linear velocity)
    # Components: base_ang_vel(3) + proj_gravity(3) + commands(3) + joint_pos(12) + joint_vel(12) + actions(12) = 45
    observation_space = 46
    
    def __post_init__(self):
        super().__post_init__()
        # Reduce number of environments for distillation training
        self.scene.num_envs = 256
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2AsymmetricPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    RSL-RL PPO configuration for asymmetric actor-critic training.
    
    Actor receives limited observations (47 dims - no linear velocity).
    Critic receives privileged observations (50 dims - includes linear velocity).
    
    RSL-RL automatically detects asymmetric observations from the environment's
    return dict: {"policy": actor_obs, "critic": critic_obs}
    """
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 50
    experiment_name = "go2_asymmetric"
    
    # Network architecture - RSL-RL will automatically size input layers
    # based on the observation dict keys from the environment
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_obs_normalization=False,  # Set to True if you want obs normalization
    #     critic_obs_normalization=False,  # Set to True if you want obs normalization
    #     actor_hidden_dims=[256, 256, 256],
    #     critic_hidden_dims=[256, 256, 256],
    #     activation="elu",
    # )
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=64,
        rnn_num_layers=2
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

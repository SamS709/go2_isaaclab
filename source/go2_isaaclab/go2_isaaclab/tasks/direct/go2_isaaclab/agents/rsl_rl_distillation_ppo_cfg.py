# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RSL-RL configuration for Go2 distillation training.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go2DistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for Go2 teacher-student distillation."""
    
    seed = 42
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "go2_distillation"
    run_name = "distillation"
    class_name = "DistillationRunner"
    
    # Observation groups mapping - tells which keys from env obs dict to use
    # For Direct RL: "policy" key goes to student, "teacher" key goes to teacher
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
    
    # These specify where to load the pre-trained teacher policy from
    # Set these to your trained asymmetric teacher checkpoint
    load_run = None  # e.g., "2024-11-24_12-30-00" - folder name of your trained asymmetric run
    load_checkpoint = None  # e.g., "model_3000.pt" or None for latest
    
    # Distillation algorithm config
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        class_name = "Distillation",
        gradient_length=5,
        learning_rate=1e-3,
        loss_type="mse",  # Mean squared error between teacher and student actions
    )
    
    # Student-Teacher policy config
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        # Required parameters
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        
        # Student network (limited observations)
        student_hidden_dims=[256, 256, 256],
        
        # Teacher network (privileged observations)
        teacher_hidden_dims=[256, 256, 256],
        
        activation="elu",
        
        # Recurrent network for student (helps compensate for missing velocity info)
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=3,
        
        # Teacher is typically not recurrent (has full state)
        teacher_recurrent=False,
    )


@configclass
class Go2StudentFinetuneRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for fine-tuning the student policy with PPO after distillation."""
    
    seed = 42
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "go2_student_finetune"
    run_name = "student_finetune"
    
    # Standard PPO algorithm
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
    # Recurrent policy (same architecture as student from distillation)
    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=0.1,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=3,
    )


@configclass
class Go2StudentPlayRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for playing trained student policies (inference mode)."""
    
    seed = 42
    num_steps_per_env = 24
    max_iterations = 0  # No training
    save_interval = 0
    experiment_name = "go2_student_play"
    run_name = "play"
    
    # Algorithm config (required even for play mode)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
    # Recurrent policy (same architecture as student from distillation)
    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=0.0,  # No exploration noise during inference
        actor_hidden_dims=[256, 256, 256],  # Match student architecture
        critic_hidden_dims=[256, 256, 256],  # Match teacher architecture
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=3,
    )

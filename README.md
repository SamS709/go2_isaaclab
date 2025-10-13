# Custom Go2 teleoperation and Sim2Sim rsl_rl training with Isaaclab

## Overview

This project/repository trains a policy for go2 unitree robot and focuses on the Sim2Sim from PhysX to Mujoco using the Newton branch of Isaaclab.

**Key Features:**

- `Train a policy` for go2 robot using direct based environnement. The policy follows the commands sent by the user: linear (x/y) velocitiezs // angular (z) velocity // base height.
- `Test` it using keyboard.
- `Sim2Sim` from PhysX to Mujoco using Newton branch of Isaaclab repo.

## Installation

- First, make sure you have the classic Isaaclab environnement installed. If you want to do the Sim2Sim, Newton Isaaclab branch is required.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
  
    ```bash
    git clone https://github.com/SamS709/go2_isaaclab.git
    ```
  
- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/go2_isaaclab
    ```

## Training a policy for PhysX

Make sure you are in your the classic Isaaclab python environnement (no Newton branch). Go in the folder where you cloned the repo.

- Train Go2:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/rsl_rl/train.py --task Isaac-Velocity-Go2-Direct-v0 --num_envs 4096 --headless
    ```

- Running trained policy :

    ```bash
    python scripts/rsl_rl/play.py --task Isaac-Velocity-Go2-Direct-v0 --num_envs 8 
    ```
- Controlling the robot with the keyboard (here, a pretrained checkpoint is used):

    ```bash
    python scripts/control/go2_locomotion.py --checkpoint pretrained_checkpoint/pretrained_checkpoint.pt
    ```

  ## Making the Sim2Sim

  Make sure to use the Newton Isaaclab python environnement. Go into the folder where you cloned the Newton branch of Isaaclab.

  
 

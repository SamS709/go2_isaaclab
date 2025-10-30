# Policy training to Real deployement for Go2 Unitree Robot

## Overview

This project/repository trains a policy for go2 unitree robot and focuses on the Sim2Sim from PhysX to Mujoco using the Newton branch of Isaaclab.

**Key Features:**

- `Train a policy` for go2 robot using direct based environnement. The policy follows the commands sent by the user: linear (x/y) velocitiezs // angular (z) velocity // base height.
- `Test` it using keyboard.
- `Sim2Sim: Newton` from PhysX to Mujoco using Newton branch of Isaaclab repo.
- `Sim2Sim: unitree_sdk_python` from PhysX to Mujoco using the unitree_mujoco repo. 

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

<img src="images/commands_control.png" width="400"/>

Controls:
- **Up/Down arrows**: Increase/decrease the robot's forward/backward velocity (x-axis)
- **Left/Right arrows**: Increase/decrease the robot's left/right velocity (y-axis) 
- **E/R keys**: Increase/decrease the robot's height (z-axis position)
- **F/G keys**: Increase/decrease the robot's angular velocity (yaw rotation)

    ```bash
    python scripts/control/go2_locomotion.py --checkpoint pretrained_checkpoint/pretrained_checkpoint.pt
    ```

## Making the Sim2Sim using Newton

Make sure to use the Newton Isaaclab python environnement. Go into the folder where you cloned the Newton branch of Isaaclab.

You need to move the files from sim2sim/newton of the repo to the Newton Isaaclab dir:
- physx_to_newton_go2.yaml into /scripts/newton_sim2sim/mappings folder
- pretrained_checkpoint.pt: create a dir named checkpoints inside scripts/newton_sim2sim/ and move the file into it.
- go2_isaaclab/ folder into source/isaaclab_tasks/isaaclab_tasks/direct/

Then run it to see the result (using newton visualizer):

<img src="images/Newton_MuJoCo.png" width="400"/>

```bash
python -m scripts.newton_sim2sim.rsl_rl_transfer \
--task Isaac-Velocity-Go2-Direct-v0 \
--num_envs 10 \
--checkpoint scripts/newton_sim2sim/checkpoints/pretrained_checkpoint.pt \
--policy_transfer_file scripts/newton_sim2sim/mappings/physx_to_newton_go2.yaml \
--newton_visualizer \
--headless
```

  
 ## Making the Sim2Sim using unitree_mujoco

Make sure to use a new python environnement (python=3.10 works fine). Make a new folder in which you clone the <a href="https://github.com/unitreerobotics/unitree_mujoco">unitree_mujoco</a> repo working with <a href = "https://github.com/unitreerobotics/unitree_sdk2_python">unitree_sdk2_python</a> (python simulator). Follow the instructions given in the repo.

Test the installation as described by the repo. launching the test_unitree_sdk2.py should look like this.

If it doesnt, and the robot stands in the air, make sure the config.py file (located in /simulate_python/config.py) looks like this (USE_JOISTICK could be set to 1 by default):

```python
ROBOT = "go2" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1" 
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene.xml" # Robot scene
DOMAIN_ID = 1 # Domain id
INTERFACE = "lo" # Interface 

USE_JOYSTICK = 0 # Simulate Unitree WirelessController using a gamepad
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False # Virtual spring band, used for lifting h1

SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer
```
You need to move the files from sim2sim/unitree_mujoco dir of the repo to the dir where you cloned unitree_mujoco github. Create a /scripts folder inside it copy the /unitree_mujoco folder (the one of this repo) in the created /scripts folder.


- Launch the Mujoco simulation (from unitree_mujoco/simulate_python) folder:

```bash
python unitree_mujoco.py
```

- Start the policy (from the unitree_mujoco/scripts/unitree_mujoco folder):

```bash
python run_policy.py
```


Then move the 
- physx_to_mujoco_go2.yaml into /scripts/newton_sim2sim/mappings folder
- pretrained_checkpoint.pt: create a dir named checkpoints inside scrimpts/newton_sim2sim/ and move the file into it.
- go2_isaaclab/ folder into source/isaaclab_tasks/isaaclab_tasks/direct/

Then run it to see the result (using newton visualizer):

<img src="images/Unitree_MuJoCo.png" width="400"/>

```bash
python -m scripts.newton_sim2sim.rsl_rl_transfer \
--task Isaac-Velocity-Go2-Direct-v0 \
--num_envs 10 \
--checkpoint scripts/newton_sim2sim/checkpoints/pretrained_checkpoint.pt \
--policy_transfer_file scripts/newton_sim2sim/mappings/physx_to_mujoco_go2.yaml \
--newton_visualizer \
--headless
```
 ## Making the Sim2Real using huro github

 pip install torch


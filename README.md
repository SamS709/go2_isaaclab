# Policy training to Real deployement for Go2 Unitree Robot

## Overview

This project/repository trains a policy for go2 unitree robot and focuses on the Sim2Sim and Sim2Real using different environments.

**Key Features:**

- `Train a policy` for go2 robot using direct based environnement. The policy follows the commands sent by the user: linear (x/y) velocitiezs // angular (z) velocity // base height.
- `Test` it using keyboard in Isaacsim.
- `Sim2Sim: Newton` from PhysX to Newton using Newton branch of Isaaclab repo.
- `Sim2Sim: unitree_mujoco` from PhysX to Mujoco using the unitree_mujoco repo. 
- `Sim2Sim: huro` sim2sim in huro environment (github of a researcher at LORIA).
- `Sim2Real: unitree_python_sdk2` sim2real in unitree_python_sdk2 using proprietary dds developed by unitree.
- `Sim2Real: huro` sim2real in huro using ros2.


## Installation


- Isaaclab sohould be installed

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
  
    ```bash
    git clone https://github.com/SamS709/go2_isaaclab.git
    ```
  
- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    cd go2_isaaclab
    python -m pip install -e source/go2_isaaclab
    ```

## Training a policy for PhysX

Make sure you are in your the classic Isaaclab python environnement (no Newton branch). Go in the folder where you cloned the repo.

- Train Go2:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/rsl_rl/train.py --task Isaac-Velocity-Go2-Asymmetric-v0 --num_envs 4096 --headless
    ```

- Running trained policy :

    ```bash
    python scripts/rsl_rl/play.py --task Isaac-Velocity-Go2-Asymmetric-v0 --num_envs 8 
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

Look at the instructions avaible [`here`](https://github.com/SamS709/go2_isaaclab_newton).


The result after Sim2sim.

<img src="images/Newton_MuJoCo.png" width="400"/>

  
 ## Making the Sim2Sim using unitree_mujoco

### Dependancies:

I highly recommand to use a conda env with python=3.10

**Install [`unitree_sdk2_python`](https://github.com/unitreerobotics/unitree_sdk2_python.git)** (follow the instructions)


**Install [`unitree_mujoco`](https://github.com/unitreerobotics/unitree_mujoco)** (follow the instructions)

In /unitree_mujoco folder, under /simulate_python/, in config.py set USE_JOYSTICK = 0 so that the robot falls on the ground:


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
```bash
pip install torch PyYAML
```

### Installation

Copy paste the /go2_mujoco folder which is in /sim2sim folder in at /unitree_mujoco root.

### Sim2sim

In one terminal (launch the simulation):
```bash
python unitree_mujoco.py 
```

In another terminal (start the policy):
```bash
python go2_publisher.py --vel-x -0.5
```


<img src="images/Unitree_MuJoCo.png" width="400"/>


 ## Making the Sim2Real using huro github

Clone the sami branch:

```bash
git clone --single-branch --branch sami https://github.com/itsikelis/huro.git
```

Follow the instructions provided in the readme of the [`github`](https://github.com/hucebot/huro/tree/sami) to see how to deploy it in sim or on the real robot.


 ## Making the Sim2Sim and Sim2Real using unitree_sdk2_python

Follow the instructions provided <a href = "https://github.com/SamS709/go2_unitree.git">here</a>





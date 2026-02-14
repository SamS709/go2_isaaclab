#!/usr/bin/env python3

"""
RL Policy Controller for Unitree Go2 Robot
Loads a PyTorch policy and controls the robot at 50Hz
"""

"""
TO RUN:
First run the simulation, then:

python go2_publisher.py --vel-x=-0.5 --policy=policy.pt --mapping=physx
python go2_publisher.py --vel-x=-0.5 --policy=policy_newton.pt --mapping=newton

Or with all parameters:

"""
import time
import torch
import os

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

from get_obs import get_obs
from mapping import Mapper

torch.set_printoptions(precision=3)


class Go2PolicyController:
    """RL Policy controller for Unitree Go2 locomotion."""

    def __init__(
        self, 
        policy_path=None, 
        vel_x=0.0,
        vel_y=0.0,
        vel_yaw=0.0,
        height=0.3,
        mapping="physx"
    ):
        """
        Initialize the policy controller.

        Args:
            policy_path: Path to the policy.pt file
            vel_x: Forward velocity command (m/s)
            vel_y: Lateral velocity command (m/s)
            vel_yaw: Yaw rate command (rad/s)
            height: Height command (m)
        """
        
        self.step_dt = 1 / 50  # policy freq = 50Hz
        self.run_policy = True

        # Load policy model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy_path = policy_path if policy_path is not None else "policy.pt"
        print(f"[INFO] Loading policy from: {policy_path}")

        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        print("[INFO] Policy loaded successfully")

        # Initialize the mapper for joint remapping
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if mapping == "physx":
            mapping_path = os.path.join(script_dir, "physx_to_mujoco_go2.yaml")
        elif mapping == "newton":
            mapping_path = os.path.join(script_dir, "newton_to_mujoco_go2.yaml")
        else:
            print("[ERROR] The specified mapping doesnt exist.")
            return
        
        default_pos_sdk = torch.tensor([
            -0.1, 0.8, -1.5,  # FR: hip, thigh, calf (actuators 0-2)
            0.1, 0.8, -1.5,  # FL: hip, thigh, calf (actuators 3-5)
            -0.1, 1.0, -1.5,  # RR: hip, thigh, calf (actuators 6-8)
            0.1, 1.0, -1.5   # RL: hip, thigh, calf (actuators 9-11)
        ])
        self.mapper = Mapper(mapping_yaml_path=mapping_path, default_pos_sdk=default_pos_sdk)
        
        # Store velocity commands
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.vel_yaw = vel_yaw
        self.height = height

        # Store latest action (for use between policy updates)
        self.current_action = torch.zeros(12)

        # Store latest messages
        self.latest_low_state = None

        # CRC for message validation
        self.crc = CRC()

        # Control parameters
        self.kp = 25.0
        self.kd = 0.5
        self.action_scale = 0.25

        # Statistics
        self.tick_count = 0
        self.start_time = time.time()

        # Initialize communication
        self.low_cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_pub.Init()

        # Subscribe to robot state
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self.low_state_callback, 10)
    
    def low_state_callback(self, msg: LowState_):
        """Store low state message."""
        self.latest_low_state = msg
        
    
    def send_motor_commands(self):
        """Send motor commands to the robot based on current action."""
        # Convert current action from policy order to SDK order
        actions_sdk_order = self.mapper.actions_policy_to_sdk(self.current_action)

        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        # Compute target positions
        target_positions = (
            self.mapper.default_pos_sdk + actions_sdk_order * self.action_scale
        )

        # Set motor commands
        for i in range(12):
            cmd.motor_cmd[i].mode = 0x01  # PMSM mode
            cmd.motor_cmd[i].q = target_positions[i]
            cmd.motor_cmd[i].kp = self.kp
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = self.kd
            cmd.motor_cmd[i].tau = 0.0

        # Calculate CRC and publish
        cmd.crc = self.crc.Crc(cmd)
        self.low_cmd_pub.Write(cmd)
    
    def run(self):
        """Main control loop."""
        print("\n" + "=" * 60)
        print("Go2 Policy Controller Running")
        print("=" * 60)
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                step_start = time.perf_counter()

                # Check if we have robot state
                if self.latest_low_state is not None:
                    self.process_control_step()
                else:
                    print("Waiting for robot state...")

                # Sleep to maintain control frequency
                time_until_next_step = self.step_dt - (
                    time.perf_counter() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("Shutting down...")
            print(f"Total ticks: {self.tick_count}")
            elapsed = time.time() - self.start_time
            print(f"Real time elapsed: {elapsed:.2f}s")
            print(
                f"Average frequency: {self.tick_count / elapsed:.1f}Hz (target: {1 / self.step_dt}Hz)"
            )
            print("=" * 60)
    
    def process_control_step(self):
        """Process one control step."""
        self.tick_count += 1
        if self.run_policy:
            self.policy_control()

    def policy_control(self):
        """Run policy inference and send motor commands."""
        # Get observation
        obs = get_obs(
            self.latest_low_state,
            self.vel_x,
            self.vel_y,
            self.vel_yaw,
            self.height,
            self.current_action,
            self.mapper,
        )

        # Run policy inference
        with torch.no_grad():
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            actions_tensor = self.policy(obs_tensor)

        actions_policy_order = actions_tensor.squeeze(0)
        self.current_action = actions_policy_order.clone()

        # Send motor commands
        self.send_motor_commands()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Go2 RL Policy Controller")
    parser.add_argument(
        "--policy", type=str, default=None, help="Path to policy.pt file"
    )
    parser.add_argument(
        "--vel-x", type=float, default=0.5, help="Forward velocity command (m/s)"
    )
    parser.add_argument(
        "--vel-y", type=float, default=0.0, help="Lateral velocity command (m/s)"
    )
    parser.add_argument(
        "--vel-yaw", type=float, default=0.0, help="Yaw rate command (rad/s)"
    )
    parser.add_argument(
        "--height", type=float, default=0.3, help="Height command (m)"
    )
    
    parser.add_argument(
        "--mapping", type=str, default="physx", help="Possible values:\n  -'physx'\n  -'newton'"
    )

    args = parser.parse_args()

    # Initialize DDS communication
    print(
        'Initializing DDS (domain_id=1, interface="lo")'
    )
    ChannelFactoryInitialize(1, "lo")

    # Create controller
    controller = Go2PolicyController(
        policy_path=args.policy,
        vel_x=args.vel_x,
        vel_y=args.vel_y,
        vel_yaw=args.vel_yaw,
        height=args.height,
        mapping=args.mapping
    )

    # Run controller
    controller.run()


if __name__ == "__main__":
    main()

"""
Configuration for Z-Axis Position Command Term

This module defines the configuration class for z-axis position command terms.
"""

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils import configclass

from .z_axis_command import ZAxisCommand


@configclass 
class ZAxisCommandCfg(CommandTermCfg):
    """Configuration for the z-axis position command generator."""

    class_type: type = ZAxisCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    goal_tolerance: float = 0.1
    """Tolerance for considering the z-axis goal as reached (in meters). Defaults to 0.1."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the z-axis position commands."""

        z_pos: tuple[float, float] = MISSING
        """Range for the z-axis position command (in meters)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the z-axis commands."""

    goal_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/z_goal"
    )
    """The configuration for the goal visualization marker. Defaults to SPHERE_MARKER_CFG."""


# Example configurations for common use cases

@configclass
class DroneAltitudeCommandCfg(ZAxisCommandCfg):
    """Configuration for drone altitude control."""
    
    asset_name: str = "robot"
    goal_tolerance: float = 0.05  # 5cm tolerance
    resampling_time_range: tuple[float, float] = (5.0, 10.0)  # Resample every 5-10 seconds
    
    @configclass
    class Ranges(ZAxisCommandCfg.Ranges):
        z_pos: tuple[float, float] = (0.5, 5.0)  # Altitude range: 0.5m to 5m
        
    ranges: Ranges = Ranges()


@configclass
class Go2JumpCommandCfg(ZAxisCommandCfg):
    """Configuration for Go2 robot jump height commands."""
    
    asset_name: str = "robot"
    goal_tolerance: float = 0.02  # 2cm tolerance
    resampling_time_range: tuple[float, float] = (2.0, 4.0)  # Resample every 2-4 seconds
    
    @configclass
    class Ranges(ZAxisCommandCfg.Ranges):
        z_pos: tuple[float, float] = (0.2, 0.8)  # Jump height range: 20cm to 80cm
        
    ranges: Ranges = Ranges()
"""Reach reset events.

The robot reset uses the Isaac Lab built-in `reset_joints_by_offset` (uniform offset around
the home pose from configs/my_so101.yaml, clamped to soft joint limits). The target is a
UniformPoseCommand, resampled by the command manager, so no custom event is needed.
`reset_joints_by_scale` (used by the official reach task) is avoided because the home
pose is all zeros and scaling zero does nothing.
"""

from isaaclab.envs.mdp import reset_joints_by_offset  # noqa: F401

__all__ = ["reset_joints_by_offset"]

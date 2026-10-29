"""ToolSandbox integration for SkillRL.

The official ToolSandbox project owns the execution backend.  This package
only adapts its scenario/trajectory contract to SkillRL's vector environment
interface, so the backend can be supplied by the installed ToolSandbox package
or by a small project-local factory.
"""

from .envs import build_toolsandbox_envs
from .projection import toolsandbox_projection

__all__ = ["build_toolsandbox_envs", "toolsandbox_projection"]

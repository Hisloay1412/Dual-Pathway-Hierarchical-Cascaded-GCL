"""
DPHCGCL core package.

Sub-packages:
    core.kinematics  — SE(3)/SO(3) 工具、POE 正逆运动学、雅可比、传输模型与迟滞
    core.gcl         — 三层 DPHCGCL 架构（传输辨识、空间 GCN、全局对齐）+ 级联逆模型
    core.monerf      — Metrology-Oriented NeRF（场、渲染、损失、位姿反演、管线）

每个子包的公共接口由各自的 __init__.py 通过 __all__ 精确控制；
本文件仅提供惰性可访问的子包句柄。
"""
from . import kinematics
from . import gcl
from . import monerf

__all__ = ["kinematics", "gcl", "monerf"]
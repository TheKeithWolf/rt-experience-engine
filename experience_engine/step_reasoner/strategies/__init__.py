"""Strategy sub-package — protocol and concrete step strategies.

Eight strategy implementations (Step 6) plus the StepStrategy protocol.
Each strategy produces a StepIntent for one cascade step type. No strategy
imports another — all shared logic lives in the injected services.
"""

from .protocol import StepStrategy
from .booster_arm import BoosterArmStrategy
from .booster_setup import BoosterSetupStrategy
from .cascade_cluster import CascadeClusterStrategy
from .initial_cluster import InitialClusterStrategy
from .initial_dead import InitialDeadStrategy
from .terminal_dead import TerminalDeadStrategy
from .terminal_near_miss import TerminalNearMissStrategy
from .wild_bridge import WildBridgeStrategy

__all__ = [
    "StepStrategy",
    "BoosterArmStrategy",
    "BoosterSetupStrategy",
    "CascadeClusterStrategy",
    "InitialClusterStrategy",
    "InitialDeadStrategy",
    "TerminalDeadStrategy",
    "TerminalNearMissStrategy",
    "WildBridgeStrategy",
]

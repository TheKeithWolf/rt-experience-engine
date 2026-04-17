"""ArchetypeFamily — single source of truth for the 11 family identifiers.

B8: replaces scattered string literals (`"dead"`, `"wild"`, `"rocket"`, …)
across the codebase with a typed StrEnum. Because StrEnum members compare
equal to their string values, YAML / JSON serialization is unchanged —
`str(ArchetypeFamily.WILD) == "wild"` and `ArchetypeFamily.WILD == "wild"`
both hold, so call sites can be migrated incrementally.
"""

from __future__ import annotations

from enum import StrEnum


class ArchetypeFamily(StrEnum):
    """The 11 archetype families recognized by the experience engine.

    Order mirrors the doc's narrative grouping: dead/t1 sit before the
    booster families (wild, rocket, bomb, lb, slb), with chain bridging
    multiple boosters and trigger / wincap / reel as feature/special
    routes. Adding a family means adding one entry here and registering
    the relevant validators / generators / archetypes — everything else
    references the enum, not the string.
    """

    DEAD = "dead"
    T1 = "t1"
    WILD = "wild"
    ROCKET = "rocket"
    BOMB = "bomb"
    LB = "lb"
    SLB = "slb"
    CHAIN = "chain"
    TRIGGER = "trigger"
    WINCAP = "wincap"
    REEL = "reel"

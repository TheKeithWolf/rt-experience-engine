"""Narrative arc and phase grammar for experience-driven cascade generation.

The narrative system replaces positional CascadeStepConstraints with a phase
grammar where each "beat" can repeat, transition predicates are data-driven,
and per-phase symbol tier constraints replace the old step-indexed tier map.
"""

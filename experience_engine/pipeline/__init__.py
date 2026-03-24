"""Pipeline — instance generation, execution, validation, and simulation.

New modules are imported directly by consumers to avoid circular import
chains through archetypes.registry → pipeline.protocols → pipeline.__init__.
"""

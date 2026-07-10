"""Backward-compatible public API for governed experiment operations.

Implementation responsibilities live in the registry, index, and transaction modules.
Callers should continue importing this façade unless they need to test an internal boundary.
"""

from .experiment_index import experiment_statuses, generate_index
from .experiment_registry import (
    CAMPAIGN_TITLES,
    CANONICAL_CAMPAIGNS,
    EXPERIMENT_CONTROL_DIR,
    EXPERIMENT_TEMPLATE_DIR,
    GENERAL_CAMPAIGN,
    REQUIRED_EXPERIMENT_FILES,
    ExperimentError,
    require_campaign,
    slugify,
    validate_registry_state,
)
from .experiment_transaction import create_experiment

__all__ = [
    "CAMPAIGN_TITLES",
    "CANONICAL_CAMPAIGNS",
    "EXPERIMENT_CONTROL_DIR",
    "EXPERIMENT_TEMPLATE_DIR",
    "GENERAL_CAMPAIGN",
    "REQUIRED_EXPERIMENT_FILES",
    "ExperimentError",
    "create_experiment",
    "experiment_statuses",
    "generate_index",
    "require_campaign",
    "slugify",
    "validate_registry_state",
]

"""
Aim experiment tracking utilities for ML training pipeline.

This module provides a lightweight wrapper around the Aim SDK to integrate
experiment tracking with minimal code changes to the training script.
"""

from utils_aim.tracker import AimTracker

__all__ = ['AimTracker']

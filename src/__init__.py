# src/ds_tools/__init__.py
"""
/*
 * Copyright (c) [2025] [Sergii Kavun]
 *
 * This software is dual-licensed:
 * - PolyForm Noncommercial 1.0.0 (default)
 * - Commercial license available
 *
 * See LICENSE for details
 */

DSTools: A library of helpful functions for various data science research stages.
"""

__version__ = "2.3.0"

from ds_tool import DSTools
from models import (
    CorrelationConfig,
    DistributionConfig,
    GrubbsTestResult,
    MetricsConfig,
    OutlierConfig,
)

__all__ = [
    "DSTools",
    "MetricsConfig",
    "CorrelationConfig",
    "OutlierConfig",
    "DistributionConfig",
    "GrubbsTestResult",
]

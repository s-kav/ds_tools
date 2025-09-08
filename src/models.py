# src/ds_tools/models.py
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


This module contains all Pydantic models used for configuration and
data validation across the ds_tools library.
"""
from typing import Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --- Configuration Models ---

class MetricsConfig(BaseModel):
    """Configuration for metrics computation."""

    error_vis: bool = Field(True, description="Flag for error visualization")
    print_values: bool = Field(False, description="Flag for printing metric values")


class CorrelationConfig(BaseModel):
    """Configuration for correlation matrix visualization."""

    font_size: int = Field(14, ge=8, le=20, description="Font size for heatmap")
    build_method: str = Field("pearson", description="Correlation method")
    image_size: Tuple[int, int] = Field((16, 16), description="Image size as tuple")

    @field_validator("build_method")
    def validate_method(cls, v: str) -> str:
        valid_methods = ["pearson", "kendall", "spearman"]
        if v not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v


class OutlierConfig(BaseModel):
    """Configuration for outlier removal."""

    sigma: float = Field(1.5, gt=0, description="IQR multiplier for outlier detection")
    change_remove: bool = Field(
        True, description="True to change outliers, False to remove"
    )
    percentage: bool = Field(True, description="Calculate outlier percentages")


class DistributionConfig(BaseModel):
    """Configuration for distribution generation."""

    mean: float = Field(description="Target mean")
    median: float = Field(description="Target median")
    std: float = Field(gt=0, description="Target standard deviation")
    min_val: float = Field(description="Minimum value")
    max_val: float = Field(description="Maximum value")
    skewness: float = Field(description="Target skewness")
    kurtosis: float = Field(description="Target kurtosis")
    n: int = Field(gt=0, description="Number of data points")
    accuracy_threshold: float = Field(
        0.01, gt=0, le=0.1, description="Accuracy threshold"
    )
    outlier_ratio: float = Field(0.025, ge=0, le=0.1, description="Proportion of outliers")

    @model_validator(mode="after")
    def validate_max_greater_than_min(self) -> "DistributionConfig":
        if self.max_val <= self.min_val:
            raise ValueError("max_val must be greater than min_val")
        return self

    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        validate_assignment = True


# --- Result Models ---

class GrubbsTestResult(BaseModel):
    """
    Pydantic model for storing the results of Grubbs' test for outliers.
    """

    is_outlier: bool = Field(
        ..., description="True if an outlier is detected, False otherwise."
    )
    g_calculated: float = Field(..., description="The calculated G-statistic for the test.")
    g_critical: float = Field(..., description="The critical G-value for the given alpha.")
    outlier_value: Optional[Union[int, float]] = Field(
        None, description="The value of the detected outlier."
    )
    outlier_index: Optional[int] = Field(None, description="The index of the detected outlier.")
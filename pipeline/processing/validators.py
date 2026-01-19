"""
Validators for pipeline data integrity.

Run these validators between stages to catch problems early.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of validation check."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self) -> bool:
        return self.valid


def validate_image_path(image_path: Optional[str], project_root: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Validate that an image path exists.
    
    Args:
        image_path: Path to image (absolute or relative)
        project_root: Root directory for resolving relative paths
        
    Returns:
        (valid, message) tuple
    """
    if image_path is None:
        return True, "No image path (text-only sample)"
    
    # Try as absolute path first
    if os.path.isabs(image_path):
        if os.path.exists(image_path):
            return True, f"Absolute path exists: {image_path}"
        return False, f"Absolute path not found: {image_path}"
    
    # Try as relative path
    if project_root:
        full_path = project_root / image_path
        if full_path.exists():
            return True, f"Relative path exists: {image_path}"
        return False, f"Relative path not found: {full_path}"
    
    # No project root, try relative to cwd
    if os.path.exists(image_path):
        return True, f"Path exists: {image_path}"
    
    return False, f"Path not found: {image_path}"


def validate_sample(sample: dict, project_root: Optional[Path] = None) -> ValidationResult:
    """
    Validate a Sample (dict or Sample object).
    
    Checks:
    1. Required fields present
    2. Image path exists (if provided)
    3. source_length matches actual length
    """
    errors = []
    warnings = []
    
    # Handle both dict and Sample object
    if hasattr(sample, 'to_dict'):
        sample = sample.to_dict()
    
    # Required fields
    required = ["id", "source_lang", "target_lang", "source_text", "source_length"]
    for field in required:
        if field not in sample:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)
    
    # Validate source_length consistency
    actual_length = len(sample["source_text"])
    if sample["source_length"] != actual_length:
        warnings.append(f"source_length mismatch: {sample['source_length']} != {actual_length}")
    
    # Validate image path
    image_path = sample.get("image_path")
    if image_path:
        valid, msg = validate_image_path(image_path, project_root)
        if not valid:
            errors.append(msg)
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def validate_samples_batch(
    samples: List[dict], 
    project_root: Optional[Path] = None,
    fail_fast: bool = False
) -> ValidationResult:
    """
    Validate a batch of samples.
    
    Args:
        samples: List of sample dicts
        project_root: Root for resolving relative paths
        fail_fast: Stop on first error if True
        
    Returns:
        Aggregated validation result
    """
    all_errors = []
    all_warnings = []
    
    for i, sample in enumerate(samples):
        result = validate_sample(sample, project_root)
        
        if result.errors:
            for e in result.errors:
                all_errors.append(f"Sample [{i}] {sample.get('id', 'unknown')}: {e}")
            if fail_fast:
                break
        
        for w in result.warnings:
            all_warnings.append(f"Sample [{i}] {sample.get('id', 'unknown')}: {w}")
    
    return ValidationResult(
        valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings
    )


def validate_prediction(prediction: dict) -> ValidationResult:
    """
    Validate a Prediction dict.
    """
    errors = []
    warnings = []
    
    # Required fields
    required = ["id", "mode", "model", "prediction", "target_lang", "inference_time_sec", "config"]
    for field in required:
        if field not in prediction:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)
    
    # Validate mode
    valid_modes = ["text_only", "text_image"]
    if prediction["mode"] not in valid_modes:
        errors.append(f"Invalid mode: {prediction['mode']}, must be one of {valid_modes}")
    
    # Check for errors in prediction
    if prediction.get("error"):
        warnings.append(f"Prediction has error: {prediction['error']}")
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


if __name__ == "__main__":
    # Quick self-test
    from pathlib import Path
    
    # Test sample validation
    good_sample = {
        "id": "test_001",
        "source_lang": "en",
        "target_lang": "zh",
        "source_text": "Hello world",
        "source_length": 11,
        "image_path": None,
        "meta": {"domain": "social"}
    }
    
    result = validate_sample(good_sample)
    print(f"Good sample valid: {result.valid}")
    
    bad_sample = {
        "id": "test_002",
        "source_lang": "en",
        # missing target_lang
        "source_text": "Hello",
        "source_length": 5
    }
    
    result = validate_sample(bad_sample)
    print(f"Bad sample valid: {result.valid}, errors: {result.errors}")

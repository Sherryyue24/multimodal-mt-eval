"""
Pipeline orchestrator - main entry point for running the evaluation pipeline.

Usage:
    python -m pipeline.run --stage all --limit 10
    python -m pipeline.run --stage inference --mode text_only
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Pipeline stage functions
from .processing.build_samples import build_samples_file, load_samples
from .processing.validators import validate_samples_batch
from .inference.text_only import run_text_only_inference
from .inference.text_image import run_text_image_inference
from .analysis.scoring import score_predictions
from .analysis.aggregation import (
    aggregate_scores_by_mode,
    aggregate_scores_by_language,
    generate_summary_report
)


def get_paths(base_dir: Path) -> Dict[str, Path]:
    """Get all pipeline file paths."""
    artifacts = base_dir / "artifacts"
    
    return {
        # Input data
        "raw_data": base_dir / "data" / "wmt2025_raw" / "wmt25.jsonl",
        
        # Samples
        "samples": artifacts / "samples" / "samples.jsonl",
        
        # Predictions
        "predictions_text_only": artifacts / "predictions" / "text_only.jsonl",
        "predictions_text_image": artifacts / "predictions" / "text_image.jsonl",
        
        # Scores
        "scores_text_only": artifacts / "scores" / "text_only_scores.jsonl",
        "scores_text_image": artifacts / "scores" / "text_image_scores.jsonl",
        
        # Summary
        "summary": artifacts / "summaries" / "summary.json",
        "manifest": artifacts / "run_manifest.json"
    }


def create_manifest(
    base_dir: Path,
    config: Dict[str, Any],
    stage_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Create run manifest for reproducibility."""
    manifest = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "stages": stage_stats,
        "paths": {k: str(v) for k, v in get_paths(base_dir).items()}
    }
    return manifest


def run_build_samples(
    paths: Dict[str, Path],
    project_root: Path,
    limit: Optional[int] = None,
    filter_has_image: bool = False
) -> Dict[str, Any]:
    """Stage 1: Build samples from raw data."""
    print("\n" + "="*60)
    print("STAGE 1: Building Samples")
    print("="*60)
    
    start_time = time.time()
    
    stats = build_samples_file(
        raw_file=paths["raw_data"],
        output_file=paths["samples"],
        project_root=project_root,
        limit=limit,
        filter_has_image=filter_has_image
    )
    
    # Validate samples
    samples = list(load_samples(paths["samples"]))
    validation = validate_samples_batch(samples, project_root)
    
    elapsed = time.time() - start_time
    
    print(f"  Total samples: {stats['total']}")
    print(f"  With images: {stats['with_image']}")
    print(f"  Validation: {'PASSED' if validation.valid else 'FAILED'}")
    if validation.warnings:
        print(f"  Warnings: {len(validation.warnings)}")
    print(f"  Time: {elapsed:.1f}s")
    
    return {
        "stats": stats,
        "validation_passed": validation.valid,
        "validation_errors": validation.errors[:5] if validation.errors else [],
        "elapsed_sec": elapsed
    }


def run_inference(
    paths: Dict[str, Path],
    mode: str,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    device: str = "mps",
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Stage 2: Run inference."""
    print("\n" + "="*60)
    print(f"STAGE 2: Inference ({mode})")
    print("="*60)
    
    start_time = time.time()
    
    if mode == "text_only":
        output_file = paths["predictions_text_only"]
        stats = run_text_only_inference(
            samples_file=paths["samples"],
            output_file=output_file,
            model_name=model_name,
            device=device,
            limit=limit
        )
    elif mode == "text_image":
        output_file = paths["predictions_text_image"]
        stats = run_text_image_inference(
            samples_file=paths["samples"],
            output_file=output_file,
            model_name=model_name,
            device=device,
            limit=limit
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    elapsed = time.time() - start_time
    
    print(f"  Total: {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Errors: {stats['error']}")
    print(f"  Time: {elapsed:.1f}s")
    
    return {
        "mode": mode,
        "stats": stats,
        "elapsed_sec": elapsed
    }


def run_scoring(
    paths: Dict[str, Path],
    mode: str
) -> Dict[str, Any]:
    """Stage 3: Score predictions."""
    print("\n" + "="*60)
    print(f"STAGE 3: Scoring ({mode})")
    print("="*60)
    
    start_time = time.time()
    
    if mode == "text_only":
        predictions_file = paths["predictions_text_only"]
        output_file = paths["scores_text_only"]
    elif mode == "text_image":
        predictions_file = paths["predictions_text_image"]
        output_file = paths["scores_text_image"]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    stats = score_predictions(
        predictions_file=predictions_file,
        samples_file=paths["samples"],
        output_file=output_file,
        metric="cometkiwi"
    )
    
    elapsed = time.time() - start_time
    
    print(f"  Scored: {stats['scored']}")
    print(f"  Avg score: {stats['avg_score']:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    
    return {
        "mode": mode,
        "stats": stats,
        "elapsed_sec": elapsed
    }


def run_aggregation(paths: Dict[str, Path]) -> Dict[str, Any]:
    """Stage 4: Aggregate results."""
    print("\n" + "="*60)
    print("STAGE 4: Aggregation")
    print("="*60)
    
    start_time = time.time()
    
    # Combine scores from both modes
    all_scores = []
    for mode in ["text_only", "text_image"]:
        scores_file = paths[f"scores_{mode}"]
        if scores_file.exists():
            all_scores.append(scores_file)
    
    if not all_scores:
        print("  No scores to aggregate!")
        return {"error": "No scores found"}
    
    # For now, aggregate the first available
    scores_by_mode = aggregate_scores_by_mode(all_scores[0])
    scores_by_lang = aggregate_scores_by_language(all_scores[0])
    
    summary = generate_summary_report(
        scores_by_mode=scores_by_mode,
        scores_by_lang=scores_by_lang,
        output_file=paths["summary"]
    )
    
    elapsed = time.time() - start_time
    
    print(f"  Summary saved to: {paths['summary']}")
    print(f"  Time: {elapsed:.1f}s")
    
    return {
        "summary": summary,
        "elapsed_sec": elapsed
    }


def run_pipeline(
    base_dir: Path,
    stages: list,
    limit: Optional[int] = None,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    device: str = "mps",
    filter_has_image: bool = False
) -> Dict[str, Any]:
    """
    Run the full pipeline or selected stages.
    
    Args:
        base_dir: Base directory (multimodal-mt-eval/)
        stages: List of stages to run
        limit: Max samples to process
        model_name: Model for inference
        device: Device for inference
        filter_has_image: Only process samples with images
    """
    paths = get_paths(base_dir)
    project_root = base_dir
    
    stage_stats = {}
    config = {
        "limit": limit,
        "model_name": model_name,
        "device": device,
        "filter_has_image": filter_has_image,
        "stages": stages
    }
    
    total_start = time.time()
    
    # Stage 1: Build samples
    if "build_samples" in stages or "all" in stages:
        stage_stats["build_samples"] = run_build_samples(
            paths, project_root, limit, filter_has_image
        )
    
    # Stage 2: Inference
    if "inference" in stages or "all" in stages:
        stage_stats["inference_text_only"] = run_inference(
            paths, "text_only", model_name, device, limit
        )
        
        # Only run text_image if we have samples with images
        samples = list(load_samples(paths["samples"]))
        has_images = any(s.image_path for s in samples)
        
        if has_images:
            stage_stats["inference_text_image"] = run_inference(
                paths, "text_image", model_name, device, limit
            )
        else:
            print("\n  Skipping text_image inference: no samples with images")
    
    # Stage 3: Scoring
    if "scoring" in stages or "all" in stages:
        if paths["predictions_text_only"].exists():
            stage_stats["scoring_text_only"] = run_scoring(paths, "text_only")
        
        if paths["predictions_text_image"].exists():
            stage_stats["scoring_text_image"] = run_scoring(paths, "text_image")
    
    # Stage 4: Aggregation
    if "aggregation" in stages or "all" in stages:
        stage_stats["aggregation"] = run_aggregation(paths)
    
    total_elapsed = time.time() - total_start
    
    # Create and save manifest
    manifest = create_manifest(base_dir, config, stage_stats)
    manifest["total_elapsed_sec"] = total_elapsed
    
    with open(paths["manifest"], 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Manifest: {paths['manifest']}")
    
    return manifest


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run multimodal MT evaluation pipeline")
    parser.add_argument(
        "--stage", 
        choices=["all", "build_samples", "inference", "scoring", "aggregation"],
        default="all",
        help="Pipeline stage to run"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Model name for inference"
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device for inference"
    )
    parser.add_argument(
        "--with-images-only",
        action="store_true",
        help="Only process samples with images"
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        # Auto-detect: look for data/wmt2025_raw/wmt25.jsonl
        script_dir = Path(__file__).parent
        base_dir = script_dir.parent  # multimodal-mt-eval/
    
    stages = [args.stage] if args.stage != "all" else ["all"]
    
    run_pipeline(
        base_dir=base_dir,
        stages=stages,
        limit=args.limit,
        model_name=args.model,
        device=args.device,
        filter_has_image=args.with_images_only
    )


if __name__ == "__main__":
    main()

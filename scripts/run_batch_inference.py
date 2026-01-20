#!/usr/bin/env python3
"""
Batch inference runner with thermal protection.

Runs inference in batches with cooling breaks between.
Follows GPT's recommendations for stability.

Usage:
    python scripts/run_batch_inference.py --mode text_only --batch-size 100
    python scripts/run_batch_inference.py --mode text_image --batch-size 100 --start 0
    python scripts/run_batch_inference.py --mode both --batch-size 100
"""

import argparse
import time
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.processing.build_samples import load_samples
from pipeline.inference.text_only import run_text_only_inference
from pipeline.inference.text_image import run_text_image_inference


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_batch(
    mode: str,
    samples_file: Path,
    output_file: Path,
    batch_size: int,
    start_idx: int,
    total_samples: int,
    cool_down_minutes: int = 5
):
    """Run a single batch of inference."""
    
    end_idx = min(start_idx + batch_size, total_samples)
    actual_batch = end_idx - start_idx
    
    print("\n" + "=" * 60)
    print(f"BATCH: {mode.upper()} | Samples {start_idx}-{end_idx} of {total_samples}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    batch_start = time.time()
    
    # Determine if we should append (resume) or overwrite
    append = start_idx > 0
    
    if mode == "text_only":
        stats = run_text_only_inference(
            samples_file=samples_file,
            output_file=output_file,
            start_idx=start_idx,
            limit=actual_batch,
            append=append
        )
    else:
        stats = run_text_image_inference(
            samples_file=samples_file,
            output_file=output_file,
            start_idx=start_idx,
            limit=actual_batch,
            append=append
        )
    
    batch_time = time.time() - batch_start
    
    print(f"\nBatch complete:")
    print(f"  Processed: {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Errors: {stats['error']}")
    print(f"  Time: {format_duration(batch_time)}")
    print(f"  Avg: {batch_time/stats['total']:.1f}s/sample" if stats['total'] > 0 else "")
    
    return stats, end_idx


def main():
    parser = argparse.ArgumentParser(description="Run batch inference with cooling breaks")
    parser.add_argument("--mode", choices=["text_only", "text_image", "both"], 
                        default="both", help="Inference mode")
    parser.add_argument("--batch-size", type=int, default=100, 
                        help="Samples per batch")
    parser.add_argument("--start", type=int, default=0,
                        help="Start from this sample index")
    parser.add_argument("--cool-down", type=int, default=5,
                        help="Minutes to wait between batches")
    parser.add_argument("--no-cool-down", action="store_true",
                        help="Skip cooling breaks (not recommended)")
    parser.add_argument("--single-batch", action="store_true",
                        help="Run only one batch then exit")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    samples_file = base_dir / "artifacts" / "samples" / "samples.jsonl"
    
    # Check samples exist
    if not samples_file.exists():
        print(f"Error: Samples file not found: {samples_file}")
        print("Run: python -m pipeline.run --stage build_samples --with-images-only")
        sys.exit(1)
    
    # Count samples
    all_samples = list(load_samples(samples_file))
    samples_with_images = [s for s in all_samples if s.image_path]
    
    print(f"\n{'='*60}")
    print("BATCH INFERENCE RUNNER")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_samples)}")
    print(f"With images: {len(samples_with_images)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Cool down: {args.cool_down} minutes" if not args.no_cool_down else "Cool down: DISABLED")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}")
    
    overall_start = time.time()
    overall_stats = {"total": 0, "success": 0, "error": 0}
    
    modes_to_run = []
    if args.mode == "both":
        modes_to_run = ["text_only", "text_image"]
    else:
        modes_to_run = [args.mode]
    
    for mode in modes_to_run:
        if mode == "text_only":
            total = len(all_samples)
            output_file = base_dir / "artifacts" / "predictions" / "text_only.jsonl"
        else:
            total = len(samples_with_images)
            output_file = base_dir / "artifacts" / "predictions" / "text_image.jsonl"
        
        current_idx = args.start
        batch_num = 0
        
        while current_idx < total:
            batch_num += 1
            
            # Cool down between batches (except first)
            if batch_num > 1 and not args.no_cool_down:
                print(f"\n⏸️  Cooling down for {args.cool_down} minutes...")
                for i in range(args.cool_down, 0, -1):
                    print(f"  {i} minutes remaining...", end="\r")
                    time.sleep(60)
                print(" " * 40)  # Clear line
            
            stats, next_idx = run_batch(
                mode=mode,
                samples_file=samples_file,
                output_file=output_file,
                batch_size=args.batch_size,
                start_idx=current_idx,
                total_samples=total
            )
            
            overall_stats["total"] += stats["total"]
            overall_stats["success"] += stats["success"]
            overall_stats["error"] += stats["error"]
            
            current_idx = next_idx
            
            if args.single_batch:
                print("\n--single-batch flag set, exiting after one batch.")
                break
        
        # Cool down between modes
        if mode != modes_to_run[-1] and not args.no_cool_down:
            print(f"\n🔄 Switching to {modes_to_run[-1]}...")
            print(f"⏸️  Cooling down for 10 minutes before next mode...")
            time.sleep(600)
    
    overall_time = time.time() - overall_start
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Total processed: {overall_stats['total']}")
    print(f"Success: {overall_stats['success']}")
    print(f"Errors: {overall_stats['error']}")
    print(f"Total time: {format_duration(overall_time)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

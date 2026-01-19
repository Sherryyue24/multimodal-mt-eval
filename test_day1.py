#!/usr/bin/env python
"""Test script for 3-sample end-to-end pipeline test."""

from pathlib import Path
import sys
import time

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline.processing.build_samples import build_samples_file, load_samples
from pipeline.processing.validators import validate_samples_batch

def test_build_samples():
    """Test sample building with image path handling."""
    raw_file = project_root / 'data' / 'wmt2025_raw' / 'wmt25.jsonl'
    output_file = project_root / 'pipeline' / 'artifacts' / 'samples' / 'test_3samples.jsonl'
    
    print('='*60)
    print('TEST 1: Building 3 samples with images')
    print('='*60)
    
    stats = build_samples_file(
        raw_file=raw_file,
        output_file=output_file,
        project_root=project_root,
        limit=3,
        filter_has_image=True
    )
    
    print(f'Stats: {stats}')
    
    # Load and display
    samples = list(load_samples(output_file))
    print(f'\nLoaded {len(samples)} samples:')
    
    all_valid = True
    for s in samples:
        print(f'  - {s.id}')
        print(f'    {s.source_lang} -> {s.target_lang}, len={s.source_length}')
        print(f'    image: {s.image_path}')
        
        if s.image_path:
            exists = Path(s.image_path).exists()
            print(f'    exists: {exists}')
            if not exists:
                all_valid = False
        print()
    
    # Validate
    print('\n' + '='*60)
    print('Validation:')
    print('='*60)
    result = validate_samples_batch(samples, project_root)
    print(f'Valid: {result.valid}')
    if result.errors:
        print(f'Errors: {result.errors}')
    if result.warnings:
        print(f'Warnings: {result.warnings}')
    
    return all_valid and result.valid, output_file


def test_text_only_inference(samples_file: Path):
    """Test text-only inference on 1 sample."""
    from pipeline.inference.text_only import TextOnlyInference
    from pipeline.processing.schemas import get_max_new_tokens, InferenceConfig
    
    print('\n' + '='*60)
    print('TEST 2: Text-only inference (1 sample)')
    print('='*60)
    
    # Load first sample
    samples = list(load_samples(samples_file))
    sample = samples[0]
    
    print(f'Sample: {sample.id}')
    print(f'Source ({sample.source_lang}): {sample.source_text[:100]}...')
    
    # Initialize inference
    engine = TextOnlyInference(device="mps")
    
    print('\nLoading model...')
    start = time.time()
    engine.load_model()
    print(f'Model loaded in {time.time() - start:.1f}s')
    
    # Run inference
    print('\nRunning inference...')
    config = InferenceConfig(
        max_new_tokens=get_max_new_tokens(sample.source_length),
        do_sample=False
    )
    
    prediction = engine.run_inference(sample, config)
    
    print(f'\nResult:')
    print(f'  Time: {prediction.inference_time_sec:.2f}s')
    print(f'  Error: {prediction.error}')
    print(f'  Target lang: {prediction.target_lang}')
    print(f'  Prediction: {prediction.prediction[:200]}...')
    
    # Cleanup
    engine.unload_model()
    
    return prediction.error is None, prediction


def test_text_image_inference(samples_file: Path):
    """Test text-image inference on 1 sample."""
    from pipeline.inference.text_image import TextImageInference
    from pipeline.processing.schemas import get_max_new_tokens, InferenceConfig
    
    print('\n' + '='*60)
    print('TEST 3: Text-image inference (1 sample)')
    print('='*60)
    
    # Load first sample with image
    samples = [s for s in load_samples(samples_file) if s.image_path]
    if not samples:
        print('No samples with images found!')
        return False, None
    
    sample = samples[0]
    
    print(f'Sample: {sample.id}')
    print(f'Source ({sample.source_lang}): {sample.source_text[:100]}...')
    print(f'Image: {sample.image_path}')
    
    # Initialize inference
    engine = TextImageInference(device="mps")
    
    print('\nLoading model...')
    start = time.time()
    engine.load_model()
    print(f'Model loaded in {time.time() - start:.1f}s')
    
    # Run inference
    print('\nRunning inference...')
    config = InferenceConfig(
        max_new_tokens=get_max_new_tokens(sample.source_length),
        do_sample=False
    )
    
    prediction = engine.run_inference(sample, config)
    
    print(f'\nResult:')
    print(f'  Time: {prediction.inference_time_sec:.2f}s')
    print(f'  Error: {prediction.error}')
    print(f'  Mode: {prediction.mode}')
    print(f'  Prediction: {prediction.prediction[:200]}...')
    
    # Cleanup
    engine.unload_model()
    
    return prediction.error is None, prediction


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['samples', 'text-only', 'text-image', 'all'], default='samples')
    args = parser.parse_args()
    
    all_passed = True
    samples_file = None
    
    # Stage 1: Build samples
    if args.stage in ['samples', 'all']:
        passed, samples_file = test_build_samples()
        all_passed = all_passed and passed
        print(f'\n{"✓ PASSED" if passed else "✗ FAILED"}')
    else:
        samples_file = project_root / 'pipeline' / 'artifacts' / 'samples' / 'test_3samples.jsonl'
    
    # Stage 2: Text-only inference
    if args.stage in ['text-only', 'all']:
        passed, _ = test_text_only_inference(samples_file)
        all_passed = all_passed and passed
        print(f'\n{"✓ PASSED" if passed else "✗ FAILED"}')
    
    # Stage 3: Text-image inference
    if args.stage in ['text-image', 'all']:
        passed, _ = test_text_image_inference(samples_file)
        all_passed = all_passed and passed
        print(f'\n{"✓ PASSED" if passed else "✗ FAILED"}')
    
    print('\n' + '='*60)
    print(f'OVERALL: {"✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"}')
    print('='*60)
    
    sys.exit(0 if all_passed else 1)

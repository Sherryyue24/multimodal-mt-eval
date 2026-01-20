#!/usr/bin/env python3
"""
Re-run inference only on failed samples.

This script:
1. Keeps successful predictions from the previous run
2. Re-runs only the failed samples with improved decoding parameters
3. Merges results into final output
"""
import json
import argparse
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.inference.text_only import TextOnlyInference
from pipeline.inference.text_image import TextImageInference
from pipeline.processing.schemas import InferenceConfig, Sample


def load_failed_ids(filepath: str) -> set:
    """Load set of failed sample IDs from file."""
    with open(filepath) as f:
        return set(line.strip() for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser(description='Re-run inference on failed samples only')
    parser.add_argument('--mode', choices=['text_only', 'text_image', 'both'], 
                        default='both', help='Inference mode')
    parser.add_argument('--failed-ids-text-only', type=str, 
                        default='artifacts/failed_ids_text_only.txt')
    parser.add_argument('--failed-ids-text-image', type=str,
                        default='artifacts/failed_ids_text_image.txt')
    parser.add_argument('--samples', type=str, 
                        default='artifacts/samples/samples.jsonl')
    parser.add_argument('--old-predictions-dir', type=str,
                        default='artifacts/predictions/old')
    parser.add_argument('--output-dir', type=str,
                        default='artifacts/predictions')
    args = parser.parse_args()
    
    # Load all samples
    samples_by_id = {}
    with open(args.samples) as f:
        for line in f:
            d = json.loads(line)
            samples_by_id[d['id']] = d
    
    original_order = list(samples_by_id.keys())
    print(f"Total samples: {len(samples_by_id)}")
    
    modes = ['text_only', 'text_image'] if args.mode == 'both' else [args.mode]
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Processing mode: {mode.upper()}")
        print(f"{'='*60}")
        
        # Load failed IDs for this mode
        failed_ids_file = args.failed_ids_text_only if mode == 'text_only' else args.failed_ids_text_image
        failed_ids = load_failed_ids(failed_ids_file)
        print(f"Failed samples to re-run: {len(failed_ids)}")
        
        # Load successful predictions from old run
        old_pred_file = Path(args.old_predictions_dir) / f"{mode}.jsonl"
        success_preds = {}
        with open(old_pred_file) as f:
            for line in f:
                pred = json.loads(line)
                if pred['id'] not in failed_ids:
                    success_preds[pred['id']] = pred
        print(f"Keeping successful predictions: {len(success_preds)}")
        
        # Initialize inference engine
        if mode == 'text_only':
            engine = TextOnlyInference()
        else:
            engine = TextImageInference()
        engine.load_model()
        
        config = InferenceConfig(max_new_tokens=256, do_sample=False)
        
        # Re-run failed samples
        new_preds = {}
        failed_samples = [samples_by_id[fid] for fid in failed_ids if fid in samples_by_id]
        total = len(failed_samples)
        
        for i, raw in enumerate(failed_samples):
            sample = Sample(
                id=raw['id'],
                source_text=raw['source_text'],
                source_lang=raw['source_lang'],
                target_lang=raw['target_lang'],
                source_length=len(raw['source_text'].split()),
                image_path=raw.get('image_path'),
                meta=raw.get('meta', {})
            )
            
            start = time.time()
            try:
                pred = engine.run_inference(sample, config)
                elapsed = time.time() - start
                
                new_preds[sample.id] = pred.to_dict()
                
                preview = pred.prediction[:60].replace('\n', ' ')
                print(f"[{i+1}/{total}] {sample.id[:35]}... ({elapsed:.1f}s) {preview}...")
                
            except Exception as e:
                print(f"[{i+1}/{total}] {sample.id} - Error: {e}")
                new_preds[sample.id] = {
                    'id': sample.id,
                    'prediction': f"[ERROR] {str(e)}",
                    'error': str(e)
                }
        
        # Merge predictions
        all_preds = {**success_preds, **new_preds}
        
        # Save in original order
        output_file = Path(args.output_dir) / f"{mode}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for sample_id in original_order:
                if sample_id in all_preds:
                    f.write(json.dumps(all_preds[sample_id], ensure_ascii=False) + '\n')
        
        print(f"\nSaved to: {output_file}")
        print(f"Total predictions: {len(all_preds)}")
        
        # Cleanup
        del engine
        import gc
        import torch
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Cool-down between modes
        if mode != modes[-1]:
            print("\nCooling down for 2 minutes...")
            time.sleep(120)
    
    print("\n" + "="*60)
    print("Re-run complete!")
    print("="*60)


if __name__ == '__main__':
    main()

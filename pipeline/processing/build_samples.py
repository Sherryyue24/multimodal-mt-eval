"""
Build standardized samples from WMT2025 raw data.

Converts WMT2025 jsonl format to pipeline Sample schema.
"""

import json
import jsonlines
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any
from dataclasses import asdict

from .schemas import Sample, get_max_new_tokens


def parse_wmt2025_record(record: Dict[str, Any], project_root: Path) -> Sample:
    """
    Convert a WMT2025 raw record to Sample schema.
    
    WMT2025 fields:
    - dataset_id, collection_id, doc_id, domain
    - src_lang, tgt_lang
    - src_text
    - video, screenshot (paths to media - folder containing images)
    - prompt_instruction
    
    Maps to Sample:
    - id: doc_id
    - source_lang: src_lang
    - target_lang: tgt_lang
    - source_text: src_text
    - source_length: len(src_text)
    - image_path: first image in screenshot folder (if not None)
    - reference_text: None (WMT2025 doesn't provide)
    - meta: domain, dataset_id, collection_id, prompt_instruction
    """
    source_text = record.get("src_text", "")
    
    # Handle image path - screenshot field points to a folder containing images
    image_path = None
    screenshot_folder = record.get("screenshot")
    if screenshot_folder:
        # Build full path to the image folder
        folder_path = project_root / "data" / "wmt2025_raw" / screenshot_folder
        if folder_path.exists() and folder_path.is_dir():
            # Find the first image file in the folder
            image_files = sorted([
                f for f in folder_path.iterdir() 
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp', '.gif']
            ])
            if image_files:
                image_path = str(image_files[0])  # Use first image
    
    return Sample(
        id=record.get("doc_id", "unknown"),
        source_lang=record.get("src_lang", "unknown"),
        target_lang=record.get("tgt_lang", "unknown"),
        source_text=source_text,
        source_length=len(source_text),
        image_path=image_path,
        reference_text=None,  # WMT2025 doesn't provide reference translations
        meta={
            "domain": record.get("domain"),
            "dataset_id": record.get("dataset_id"),
            "collection_id": record.get("collection_id"),
            "prompt_instruction": record.get("prompt_instruction"),
            "video": record.get("video"),  # Keep for potential future use
        }
    )


def load_wmt2025_raw(
    raw_file: Path, 
    project_root: Path,
    limit: Optional[int] = None,
    filter_has_image: bool = False
) -> Iterator[Sample]:
    """
    Load WMT2025 raw data and yield Sample objects.
    
    Args:
        raw_file: Path to wmt25.jsonl
        project_root: Project root for resolving relative paths
        limit: Maximum number of samples to load (for testing)
        filter_has_image: If True, only yield samples with images
        
    Yields:
        Sample objects
    """
    count = 0
    with jsonlines.open(raw_file) as reader:
        for record in reader:
            sample = parse_wmt2025_record(record, project_root)
            
            if filter_has_image and sample.image_path is None:
                continue
            
            yield sample
            count += 1
            
            if limit and count >= limit:
                break


def build_samples_file(
    raw_file: Path,
    output_file: Path,
    project_root: Path,
    limit: Optional[int] = None,
    filter_has_image: bool = False,
    target_langs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Build samples file from WMT2025 raw data.
    
    Args:
        raw_file: Input wmt25.jsonl path
        output_file: Output samples.jsonl path
        project_root: Project root for path resolution
        limit: Max samples to process
        filter_has_image: Only include samples with images
        target_langs: Filter to specific target languages
        
    Returns:
        Stats dict with counts
    """
    stats = {
        "total": 0,
        "with_image": 0,
        "by_target_lang": {},
        "by_domain": {}
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(output_file, mode='w') as writer:
        for sample in load_wmt2025_raw(raw_file, project_root, limit=None, filter_has_image=filter_has_image):
            
            # Apply target language filter
            if target_langs and sample.target_lang not in target_langs:
                continue
            
            # Apply limit
            if limit and stats["total"] >= limit:
                break
            
            # Write sample
            writer.write(sample.to_dict())
            
            # Update stats
            stats["total"] += 1
            if sample.image_path:
                stats["with_image"] += 1
            
            tl = sample.target_lang
            stats["by_target_lang"][tl] = stats["by_target_lang"].get(tl, 0) + 1
            
            domain = sample.meta.get("domain", "unknown")
            stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1
    
    return stats


def load_samples(samples_file: Path) -> Iterator[Sample]:
    """
    Load samples from a samples.jsonl file.
    
    Args:
        samples_file: Path to samples.jsonl
        
    Yields:
        Sample objects
    """
    with jsonlines.open(samples_file) as reader:
        for record in reader:
            yield Sample.from_dict(record)


def select_samples_for_testing(
    samples: List[Sample],
    n_per_lang: int = 1,
    require_image: bool = False
) -> List[Sample]:
    """
    Select a diverse subset of samples for testing.
    
    Selects n samples per target language, prioritizing samples with images.
    
    Args:
        samples: List of all samples
        n_per_lang: Number of samples per target language
        require_image: Only select samples with images
        
    Returns:
        Selected samples list
    """
    by_lang: Dict[str, List[Sample]] = {}
    
    for sample in samples:
        tl = sample.target_lang
        if tl not in by_lang:
            by_lang[tl] = []
        by_lang[tl].append(sample)
    
    selected = []
    for lang, lang_samples in by_lang.items():
        # Sort: images first, then by source length variety
        if require_image:
            lang_samples = [s for s in lang_samples if s.image_path]
        else:
            # Prioritize samples with images
            lang_samples.sort(key=lambda s: (s.image_path is None, s.source_length))
        
        selected.extend(lang_samples[:n_per_lang])
    
    return selected


if __name__ == "__main__":
    # Test the builder
    import sys
    
    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # multimodal-mt-eval/
    
    raw_file = project_root / "data" / "wmt2025_raw" / "wmt25.jsonl"
    output_file = script_dir.parent / "artifacts" / "samples" / "test_samples.jsonl"
    
    print(f"Project root: {project_root}")
    print(f"Raw file: {raw_file}")
    print(f"Output file: {output_file}")
    
    if not raw_file.exists():
        print(f"Error: Raw file not found: {raw_file}")
        sys.exit(1)
    
    # Build samples with limit for testing
    stats = build_samples_file(
        raw_file=raw_file,
        output_file=output_file,
        project_root=project_root,
        limit=10,  # Just 10 for testing
        filter_has_image=False
    )
    
    print(f"\nBuild stats:")
    print(f"  Total samples: {stats['total']}")
    print(f"  With image: {stats['with_image']}")
    print(f"  By target lang: {stats['by_target_lang']}")
    print(f"  By domain: {stats['by_domain']}")
    
    # Test loading
    print(f"\nLoading samples...")
    samples = list(load_samples(output_file))
    print(f"Loaded {len(samples)} samples")
    
    if samples:
        print(f"\nFirst sample:")
        print(f"  ID: {samples[0].id}")
        print(f"  {samples[0].source_lang} -> {samples[0].target_lang}")
        print(f"  Text length: {samples[0].source_length}")
        print(f"  Has image: {samples[0].image_path is not None}")

#!/usr/bin/env python3
"""
Error Taxonomy Analysis (AUXILIARY - not a replacement for required metrics)

This module performs error/failure analysis on translation outputs:
- Detects: repetition, language mixing, prompt leakage, no translation
- Compares error rates between text-only and text-image modes
- Useful for understanding WHY one mode outperforms another

NOTE: This is AUXILIARY analysis for debugging/explanation purposes.
      The REQUIRED evaluation metrics are:
      1. CometKiwi (MT quality score)
      2. LLM-as-a-Judge (pairwise comparison)

Usage:
    # As module
    from pipeline.analysis.error_taxonomy import run_error_analysis
    stats = run_error_analysis(samples_file, to_preds, ti_preds, output_dir)
    
    # As script
    python -m pipeline.analysis.error_taxonomy
"""
import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional


def detect_issues(pred: str, sample_id: str, source_text: Optional[str] = None) -> List[str]:
    """
    Detect quality issues in a prediction.
    
    Args:
        pred: The model's prediction text
        sample_id: Sample identifier (contains language info)
        source_text: Original source text (for checking if mixing is from source)
    
    Returns:
        List of detected issue types
    """
    issues = []
    
    # Repetition detection
    if '!!!' in pred or (len(pred) >= 50 and len(set(pred[:50])) < 8):
        issues.append('repetition')
    
    # Prompt leakage detection
    if 'Output ONLY' in pred or 'translation accuracy' in pred.lower():
        issues.append('prompt_leak')
    
    # Language mixing: Chinese characters in non-Chinese translations
    if '_zh_' not in sample_id and '_cmn_' not in sample_id:
        if re.search(r'[\u4e00-\u9fff]', pred):
            # Check if source text also contains Chinese (not an error then)
            if source_text and re.search(r'[\u4e00-\u9fff]', source_text):
                pass  # Source has Chinese, not counting as mixing
            else:
                issues.append('lang_mixing')
    
    return issues


def run_error_analysis(
    samples_file: Path,
    text_only_preds: Path,
    text_image_preds: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run error taxonomy analysis on predictions.
    
    Args:
        samples_file: Path to samples.jsonl
        text_only_preds: Path to text_only predictions
        text_image_preds: Path to text_image predictions
        output_dir: Optional directory to save reports
        verbose: Whether to print detailed output
        
    Returns:
        Dict with analysis statistics
    """
    # Load samples
    samples = {}
    with open(samples_file) as f:
        for line in f:
            d = json.loads(line)
            samples[d['id']] = d
    
    # Load predictions
    preds_to = {}
    preds_ti = {}
    
    with open(text_only_preds) as f:
        for line in f:
            d = json.loads(line)
            preds_to[d['id']] = d
    
    with open(text_image_preds) as f:
        for line in f:
            d = json.loads(line)
            preds_ti[d['id']] = d
    
    # Per-language statistics
    lang_stats = defaultdict(lambda: {
        'count': 0,
        'to_issues': 0,
        'ti_issues': 0,
        'to_length': 0,
        'ti_length': 0,
        'ti_better': 0,
        'to_better': 0,
    })
    
    # Per-domain statistics
    domain_stats = defaultdict(lambda: {
        'count': 0,
        'to_issues': 0,
        'ti_issues': 0,
    })
    
    # Issue type counts
    issue_types_to = defaultdict(int)
    issue_types_ti = defaultdict(int)
    
    # Sample-by-sample comparison details
    comparison_details = []
    
    for sample_id, sample in samples.items():
        if sample_id not in preds_to or sample_id not in preds_ti:
            continue
        
        pred_to = preds_to[sample_id].get('prediction', '')
        pred_ti = preds_ti[sample_id].get('prediction', '')
        source_text = sample.get('source_text', '')
        
        # Extract language pair from sample ID
        parts = sample_id.split('#')
        if len(parts) >= 2:
            lang_pair = parts[0].strip('-_')
        else:
            lang_pair = 'unknown'
        
        # Extract domain
        domain = sample.get('meta', {}).get('domain', 'unknown')
        if not domain:
            domain = 'social' if 'social' in sample_id else 'speech'
        
        # Detect issues in both predictions
        issues_to = detect_issues(pred_to, sample_id, source_text)
        issues_ti = detect_issues(pred_ti, sample_id, source_text)
        
        # Count issue types
        for issue in issues_to:
            issue_types_to[issue] += 1
        for issue in issues_ti:
            issue_types_ti[issue] += 1
        
        # Update language statistics
        lang_stats[lang_pair]['count'] += 1
        lang_stats[lang_pair]['to_issues'] += 1 if issues_to else 0
        lang_stats[lang_pair]['ti_issues'] += 1 if issues_ti else 0
        lang_stats[lang_pair]['to_length'] += len(pred_to)
        lang_stats[lang_pair]['ti_length'] += len(pred_ti)
        
        # Compare which mode performed better
        if len(issues_ti) < len(issues_to):
            lang_stats[lang_pair]['ti_better'] += 1
        elif len(issues_to) < len(issues_ti):
            lang_stats[lang_pair]['to_better'] += 1
        
        # Update domain statistics
        domain_stats[domain]['count'] += 1
        domain_stats[domain]['to_issues'] += 1 if issues_to else 0
        domain_stats[domain]['ti_issues'] += 1 if issues_ti else 0
        
        # Record detailed comparison for samples with different outcomes
        if issues_to != issues_ti:
            comparison_details.append({
                'id': sample_id,
                'lang_pair': lang_pair,
                'to_issues': issues_to,
                'ti_issues': issues_ti,
                'to_preview': pred_to[:80] if pred_to else '',
                'ti_preview': pred_ti[:80] if pred_ti else '',
            })
    
    # Calculate totals
    total_to_issues = sum(s['to_issues'] for s in lang_stats.values())
    total_ti_issues = sum(s['ti_issues'] for s in lang_stats.values())
    total_count = sum(s['count'] for s in lang_stats.values())
    
    # Build result
    result = {
        'total_samples': total_count,
        'text_only_issues': total_to_issues,
        'text_image_issues': total_ti_issues,
        'improvement': total_to_issues - total_ti_issues,
        'improvement_pct': (total_to_issues - total_ti_issues) / total_count * 100 if total_count else 0,
        'issue_types': {
            'text_only': dict(issue_types_to),
            'text_image': dict(issue_types_ti),
        },
        'per_language': {k: dict(v) for k, v in lang_stats.items()},
        'per_domain': {k: dict(v) for k, v in domain_stats.items()},
    }
    
    # Save reports if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON data
        data_file = output_dir / 'error_taxonomy_data.json'
        with open(data_file, 'w') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                **result,
                'comparison_details': comparison_details[:50],  # Top 50
            }, f, indent=2, ensure_ascii=False)
        
        # Generate and save text report
        report_lines = _generate_report_lines(result, lang_stats, domain_stats, comparison_details)
        report_file = output_dir / 'error_taxonomy_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        if verbose:
            print('\n'.join(report_lines))
    
    return result


def _generate_report_lines(
    result: Dict[str, Any],
    lang_stats: Dict,
    domain_stats: Dict,
    comparison_details: List[Dict]
) -> List[str]:
    """Generate text report lines."""
    lines = []
    
    lines.append("=" * 70)
    lines.append("ERROR TAXONOMY ANALYSIS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    
    lines.append("\n### 1. Overall Quality Comparison")
    lines.append("-" * 50)
    lines.append(f"Total samples: {result['total_samples']}")
    lines.append(f"Text-Only problematic samples: {result['text_only_issues']} "
                f"({result['text_only_issues']/result['total_samples']*100:.1f}%)")
    lines.append(f"Text-Image problematic samples: {result['text_image_issues']} "
                f"({result['text_image_issues']/result['total_samples']*100:.1f}%)")
    lines.append(f"Improvement: {result['improvement']} samples ({result['improvement_pct']:.1f}%)")
    
    lines.append("\n### 2. Issue Types")
    lines.append("-" * 50)
    lines.append(f"Text-Only:  {dict(result['issue_types']['text_only'])}")
    lines.append(f"Text-Image: {dict(result['issue_types']['text_image'])}")
    
    lines.append("\n### 3. Per-Language Analysis")
    lines.append("-" * 70)
    lines.append(f"{'Lang Pair':<20} {'Count':<8} {'TO Err%':<10} {'TI Err%':<10} {'TI Win':<8} {'TO Win':<8}")
    lines.append("-" * 70)
    
    for lang, stats in sorted(lang_stats.items(), key=lambda x: x[1]['count'], reverse=True):
        n = stats['count']
        to_pct = stats['to_issues'] / n * 100 if n > 0 else 0
        ti_pct = stats['ti_issues'] / n * 100 if n > 0 else 0
        lines.append(f"{lang:<20} {n:<8} {to_pct:<10.1f} {ti_pct:<10.1f} {stats['ti_better']:<8} {stats['to_better']:<8}")
    
    lines.append("\n### 4. Per-Domain Analysis")
    lines.append("-" * 50)
    for domain, stats in sorted(domain_stats.items()):
        n = stats['count']
        to_pct = stats['to_issues'] / n * 100 if n > 0 else 0
        ti_pct = stats['ti_issues'] / n * 100 if n > 0 else 0
        lines.append(f"{domain}: TO errors={to_pct:.1f}%, TI errors={ti_pct:.1f}% (n={n})")
    
    lines.append("\n### 5. Sample Differences (Top 10)")
    lines.append("-" * 70)
    for detail in comparison_details[:10]:
        lines.append(f"\nSample: {detail['id'][:50]}...")
        lines.append(f"  TO issues: {detail['to_issues'] or 'None'}")
        lines.append(f"  TI issues: {detail['ti_issues'] or 'None'}")
    
    lines.append("\n" + "=" * 70)
    
    return lines


def main():
    """CLI entry point."""
    base_dir = Path(__file__).parent.parent.parent  # pipeline/ -> multimodal-mt-eval/
    
    samples_file = base_dir / 'artifacts/samples/samples.jsonl'
    text_only_preds = base_dir / 'artifacts/predictions/text_only.jsonl'
    text_image_preds = base_dir / 'artifacts/predictions/text_image.jsonl'
    output_dir = base_dir / 'artifacts/reports'
    
    result = run_error_analysis(
        samples_file=samples_file,
        text_only_preds=text_only_preds,
        text_image_preds=text_image_preds,
        output_dir=output_dir,
        verbose=True
    )
    
    print(f"\nJSON data saved to: {output_dir / 'error_taxonomy_data.json'}")
    print(f"Text report saved to: {output_dir / 'error_taxonomy_report.txt'}")


if __name__ == '__main__':
    main()

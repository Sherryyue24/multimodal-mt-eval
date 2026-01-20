#!/usr/bin/env python3
"""
Identify failed samples for targeted re-inference.

Failure types detected:
1. repetition - Output contains repetitive patterns
2. prompt_leak - Prompt instructions leaked into output
3. no_translation - Output is just language code, no actual translation
4. lang_mixing - Chinese characters in non-Chinese translations
"""
import json
import re
import argparse
from pathlib import Path

# Chinese character detection pattern
CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fff]')

# Keywords indicating prompt leakage
PROMPT_LEAK_KEYWORDS = [
    'Output ONLY',
    'USE ANY',
    'translation accuracy',
    'Translate the following',
    'Do not include',
    'context only if',
]

# Patterns indicating no actual translation (just language codes)
NO_TRANSLATION_PATTERNS = [
    r'^Bho[\s_]?IN\s*$',
    r'^ar[\s_]?EG\s*$',
    r'^bn[\s_]?BD\s*$',
    r'^ja[\s_]?JP\s*$',
    r'^zh[\s_]?CN\s*$',
]


def detect_repetition(text: str, threshold: float = 0.3) -> bool:
    """
    Detect repetition issues using 4-gram analysis.
    
    Args:
        text: The prediction text to analyze
        threshold: Maximum allowed repetition ratio (0.3 = 30% repeated)
    
    Returns:
        True if repetition is detected
    """
    if len(text) < 20:
        return False
    
    # Obvious repetition patterns
    if '!!!' in text or '🛠🔧🛠' in text:
        return True
    
    # Low character diversity
    if len(text) >= 50 and len(set(text[:50])) < 8:
        return True
    
    # 4-gram repetition detection
    words = text.split()
    if len(words) < 8:
        return False
    
    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    if not ngrams:
        return False
    
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return unique_ratio < (1 - threshold)


def detect_prompt_leak(text: str) -> bool:
    """Detect if prompt instructions leaked into output."""
    text_lower = text.lower()
    for keyword in PROMPT_LEAK_KEYWORDS:
        if keyword.lower() in text_lower:
            return True
    return False


def detect_no_translation(text: str) -> bool:
    """Detect if output is just a language code without actual translation."""
    text_stripped = text.strip()
    for pattern in NO_TRANSLATION_PATTERNS:
        if re.match(pattern, text_stripped, re.IGNORECASE):
            return True
    return len(text_stripped) < 3


def detect_lang_mixing(text: str, sample_id: str, source_text: str = None) -> bool:
    """
    Detect language mixing (Chinese in non-Chinese translations).
    
    Note: If source text contains Chinese, it's not counted as an error
    (could be intentional preservation).
    """
    # Only check for non-Chinese target languages
    if '_zh_' in sample_id or '_cmn_' in sample_id:
        return False
    
    has_chinese = bool(CHINESE_PATTERN.search(text))
    
    if not has_chinese:
        return False
    
    # If source also has Chinese, might be intentional (not an error)
    if source_text and CHINESE_PATTERN.search(source_text):
        return False
    
    return True


def analyze_sample(prediction: str, sample_id: str, source_text: str = None) -> dict:
    """
    Analyze a single sample for quality issues.
    
    Returns:
        Dict with 'has_issues' bool and 'issues' list
    """
    issues = []
    
    if detect_repetition(prediction):
        issues.append('repetition')
    
    if detect_prompt_leak(prediction):
        issues.append('prompt_leak')
    
    if detect_no_translation(prediction):
        issues.append('no_translation')
    
    if detect_lang_mixing(prediction, sample_id, source_text):
        issues.append('lang_mixing')
    
    return {
        'has_issues': len(issues) > 0,
        'issues': issues
    }


def main():
    parser = argparse.ArgumentParser(description='Identify failed samples for re-inference')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions JSONL file')
    parser.add_argument('--samples', type=str, default=None,
                        help='Path to samples JSONL file (for source text checking)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for failed sample IDs')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')
    args = parser.parse_args()
    
    # Load samples if provided
    source_texts = {}
    if args.samples and Path(args.samples).exists():
        with open(args.samples) as f:
            for line in f:
                d = json.loads(line)
                source_texts[d['id']] = d.get('source_text', '')
    
    # Analyze predictions
    with open(args.predictions) as f:
        predictions = [json.loads(line) for line in f]
    
    failed_ids = []
    issue_counts = {
        'repetition': 0,
        'prompt_leak': 0,
        'no_translation': 0,
        'lang_mixing': 0
    }
    
    for pred in predictions:
        sample_id = pred['id']
        prediction_text = pred.get('prediction', '')
        source_text = source_texts.get(sample_id, None)
        
        result = analyze_sample(prediction_text, sample_id, source_text)
        
        if result['has_issues']:
            failed_ids.append(sample_id)
            for issue in result['issues']:
                issue_counts[issue] += 1
            
            if args.verbose and len(failed_ids) <= 5:
                print(f"Failed: {sample_id}")
                print(f"  Issues: {result['issues']}")
                print(f"  Preview: {prediction_text[:80]}...")
                print()
    
    # Save failed IDs
    with open(args.output, 'w') as f:
        for sample_id in failed_ids:
            f.write(sample_id + '\n')
    
    # Print summary report
    total = len(predictions)
    failed = len(failed_ids)
    print(f"\n{'='*50}")
    print(f"Failed Sample Analysis Report")
    print(f"{'='*50}")
    print(f"Total samples: {total}")
    print(f"Failed samples: {failed} ({failed/total*100:.1f}%)")
    print(f"Passed samples: {total - failed} ({(total-failed)/total*100:.1f}%)")
    print(f"\nBy issue type:")
    for issue, count in issue_counts.items():
        print(f"  {issue}: {count} ({count/total*100:.1f}%)")
    print(f"\nFailed sample IDs saved to: {args.output}")


if __name__ == '__main__':
    main()

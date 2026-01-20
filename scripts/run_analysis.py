#!/usr/bin/env python3
"""计算 BERTScore 并生成分析报告"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    base_dir = Path(__file__).parent.parent
    
    # 加载样本
    samples = {}
    with open(base_dir / 'artifacts/samples/samples.jsonl') as f:
        for line in f:
            d = json.loads(line)
            samples[d['id']] = d
    
    # 加载预测
    predictions_to = []
    predictions_ti = []
    sample_ids = []
    references = []
    
    with open(base_dir / 'artifacts/predictions/text_only.jsonl') as f:
        for line in f:
            d = json.loads(line)
            predictions_to.append(d['prediction'])
            sample_ids.append(d['id'])
            ref = samples[d['id']].get('reference_text', '')
            references.append(ref if ref else '')
    
    with open(base_dir / 'artifacts/predictions/text_image.jsonl') as f:
        for line in f:
            d = json.loads(line)
            predictions_ti.append(d['prediction'])
    
    print(f"总样本数: {len(predictions_to)}")
    
    # 检查参考译文
    valid_indices = [i for i, r in enumerate(references) if r and r.strip()]
    print(f"有参考译文的样本: {len(valid_indices)}")
    
    if len(valid_indices) == 0:
        print("\n没有参考译文，使用无参考评估方法")
        print("计算预测长度和字符多样性作为质量指标...")
        
        # 无参考评估：比较两个模式的输出特征
        import re
        
        def analyze_output(pred):
            """分析输出质量"""
            length = len(pred)
            word_count = len(pred.split())
            unique_chars = len(set(pred))
            has_repetition = '!!!' in pred or len(set(pred[:50])) < 8 if len(pred) >= 50 else False
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', pred))
            return {
                'length': length,
                'word_count': word_count,
                'unique_chars': unique_chars,
                'has_repetition': has_repetition,
                'has_chinese': has_chinese
            }
        
        stats_to = {'length': 0, 'word_count': 0, 'unique_chars': 0, 'repetition': 0, 'chinese': 0}
        stats_ti = {'length': 0, 'word_count': 0, 'unique_chars': 0, 'repetition': 0, 'chinese': 0}
        
        for i in range(len(predictions_to)):
            a_to = analyze_output(predictions_to[i])
            a_ti = analyze_output(predictions_ti[i])
            
            stats_to['length'] += a_to['length']
            stats_to['word_count'] += a_to['word_count']
            stats_to['unique_chars'] += a_to['unique_chars']
            stats_to['repetition'] += 1 if a_to['has_repetition'] else 0
            stats_to['chinese'] += 1 if a_to['has_chinese'] else 0
            
            stats_ti['length'] += a_ti['length']
            stats_ti['word_count'] += a_ti['word_count']
            stats_ti['unique_chars'] += a_ti['unique_chars']
            stats_ti['repetition'] += 1 if a_ti['has_repetition'] else 0
            stats_ti['chinese'] += 1 if a_ti['has_chinese'] else 0
        
        n = len(predictions_to)
        
        print("\n" + "="*60)
        print("输出质量对比 (无参考评估)")
        print("="*60)
        print(f"{'指标':<20} {'Text-Only':<15} {'Text-Image':<15} {'差异':<10}")
        print("-"*60)
        print(f"{'平均长度 (字符)':<20} {stats_to['length']/n:<15.1f} {stats_ti['length']/n:<15.1f} {(stats_ti['length']-stats_to['length'])/n:+.1f}")
        print(f"{'平均词数':<20} {stats_to['word_count']/n:<15.1f} {stats_ti['word_count']/n:<15.1f} {(stats_ti['word_count']-stats_to['word_count'])/n:+.1f}")
        print(f"{'平均唯一字符数':<20} {stats_to['unique_chars']/n:<15.1f} {stats_ti['unique_chars']/n:<15.1f} {(stats_ti['unique_chars']-stats_to['unique_chars'])/n:+.1f}")
        print(f"{'重复问题样本数':<20} {stats_to['repetition']:<15} {stats_ti['repetition']:<15} {stats_ti['repetition']-stats_to['repetition']:+d}")
        print(f"{'中文混入样本数':<20} {stats_to['chinese']:<15} {stats_ti['chinese']:<15} {stats_ti['chinese']-stats_to['chinese']:+d}")
        
    else:
        print(f"\n使用 {len(valid_indices)} 个有参考的样本计算 BERTScore...")
        
        preds_to = [predictions_to[i] for i in valid_indices]
        preds_ti = [predictions_ti[i] for i in valid_indices]
        refs = [references[i] for i in valid_indices]
        
        from bert_score import score
        
        print("\n计算 Text-Only BERTScore...")
        P_to, R_to, F1_to = score(preds_to, refs, lang='multilingual', verbose=True)
        
        print("\n计算 Text-Image BERTScore...")
        P_ti, R_ti, F1_ti = score(preds_ti, refs, lang='multilingual', verbose=True)
        
        print("\n" + "="*60)
        print("BERTScore 结果")
        print("="*60)
        print(f"Text-Only:  P={P_to.mean():.4f}, R={R_to.mean():.4f}, F1={F1_to.mean():.4f}")
        print(f"Text-Image: P={P_ti.mean():.4f}, R={R_ti.mean():.4f}, F1={F1_ti.mean():.4f}")
        
        diff = F1_ti.mean() - F1_to.mean()
        pct = diff / F1_to.mean() * 100
        print(f"\n差异: {diff:.4f} ({pct:+.1f}%)")
        
        if diff > 0:
            print("结论: Text-Image 模式表现更好")
        elif diff < 0:
            print("结论: Text-Only 模式表现更好")
        else:
            print("结论: 两种模式表现相当")


if __name__ == '__main__':
    main()

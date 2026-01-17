"""
Tests for evaluation metrics.
"""

import pytest
from multimodal_mt_eval.metrics import BLEUMetric, BERTScoreMetric


class TestBLEUMetric:
    """Tests for BLEU metric."""
    
    def test_bleu_perfect_match(self):
        """Test BLEU with perfect match."""
        metric = BLEUMetric()
        predictions = ["Hello world"]
        references = ["Hello world"]
        
        score = metric.compute(predictions, references)
        assert score == 100.0
    
    def test_bleu_no_match(self):
        """Test BLEU with no match."""
        metric = BLEUMetric()
        predictions = ["Completely different text"]
        references = ["Something else entirely"]
        
        score = metric.compute(predictions, references)
        assert score == 0.0
    
    def test_bleu_partial_match(self):
        """Test BLEU with partial match."""
        metric = BLEUMetric()
        predictions = ["The cat sits on the mat"]
        references = ["A cat is sitting on a mat"]
        
        score = metric.compute(predictions, references)
        assert 0 < score < 100


class TestBERTScoreMetric:
    """Tests for BERTScore metric."""
    
    def test_bertscore_similar_sentences(self):
        """Test BERTScore with semantically similar sentences."""
        metric = BERTScoreMetric(device="cpu")
        predictions = ["A dog is running in the park"]
        references = ["The dog runs in the park"]
        
        score = metric.compute(predictions, references)
        assert 0.8 < score <= 1.0
    
    def test_bertscore_different_sentences(self):
        """Test BERTScore with different sentences."""
        metric = BERTScoreMetric(device="cpu")
        predictions = ["The weather is nice today"]
        references = ["I enjoy eating pizza"]
        
        score = metric.compute(predictions, references)
        assert 0 < score < 0.7

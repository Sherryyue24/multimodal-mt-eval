"""
Tests for the main evaluator.
"""

import pytest
from multimodal_mt_eval import MultimodalMTEvaluator


class TestMultimodalMTEvaluator:
    """Tests for MultimodalMTEvaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = MultimodalMTEvaluator(metrics=["bleu"], device="cpu")
        assert "bleu" in evaluator.metrics
        assert "bleu" in evaluator.metric_objects
    
    def test_evaluate(self):
        """Test basic evaluation."""
        evaluator = MultimodalMTEvaluator(metrics=["bleu"], device="cpu")
        
        predictions = ["Hello world", "Good morning"]
        references = ["Hello world", "Good morning"]
        
        results = evaluator.evaluate(predictions, references)
        
        assert "bleu" in results
        assert isinstance(results["bleu"], float)
        assert results["bleu"] == 100.0
    
    def test_evaluate_batch(self):
        """Test batch evaluation."""
        evaluator = MultimodalMTEvaluator(metrics=["bleu"], device="cpu")
        
        data = [
            {"prediction": "Hello", "reference": "Hello"},
            {"prediction": "World", "reference": "World"},
        ]
        
        results = evaluator.evaluate_batch(data)
        
        assert "overall" in results
        assert "num_examples" in results
        assert results["num_examples"] == 2
        assert "bleu" in results["overall"]

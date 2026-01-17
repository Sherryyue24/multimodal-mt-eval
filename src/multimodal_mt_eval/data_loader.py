"""
Data loading utilities for multimodal machine translation datasets.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from PIL import Image


class MultimodalDataLoader:
    """
    Data loader for multimodal machine translation datasets.
    
    Supports loading data from various formats including JSON, CSV, and
    custom multimodal dataset formats.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the dataset
        """
        self.data_dir = Path(data_dir) if data_dir else None
    
    def load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to JSON file
        
        Returns:
            List of data examples
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def load_multimodal_dataset(
        self,
        text_file: str,
        image_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load a multimodal dataset with text and images.
        
        Args:
            text_file: Path to file containing text data (JSON/JSONL)
            image_dir: Directory containing images
        
        Returns:
            List of examples with text and image data
        """
        # Load text data
        if text_file.endswith('.json'):
            with open(text_file, 'r', encoding='utf-8') as f:
                text_data = json.load(f)
        elif text_file.endswith('.jsonl'):
            text_data = []
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    text_data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {text_file}")
        
        # Load images if directory is provided
        if image_dir:
            image_path = Path(image_dir)
            for example in text_data:
                if 'image_id' in example or 'image_file' in example:
                    img_file = example.get('image_file') or f"{example['image_id']}.jpg"
                    img_path = image_path / img_file
                    if img_path.exists():
                        example['image'] = Image.open(img_path)
        
        return text_data
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_file: Path to output file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

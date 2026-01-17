"""
Download WMT2025 multimodal MT data from public sources.
This script handles both text data (JSONL) and image assets (ZIP).
"""

import os
import json
import zipfile
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm


# Download URLs (public, no authentication required)
URLS = {
    "text": "https://data.statmt.org/wmt25/general-mt/wmt25.jsonl",
    "images": "https://data.statmt.org/wmt25/general-mt/wmt25_genmt_assets.zip"
}


def download_file(url: str, output_path: str, description: str = "Downloading") -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: File URL
        output_path: Local save path
        description: Progress bar description
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n📥 {description}")
        print(f"   URL: {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=Path(output_path).name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """
    Extract zip file.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n📦 Extracting: {Path(zip_path).name}")
        
        Path(extract_to).mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"✅ Extracted to: {extract_to}")
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def verify_data(data_dir: str) -> bool:
    """
    Verify downloaded data integrity.
    
    Args:
        data_dir: Directory containing the data
        
    Returns:
        True if data looks valid, False otherwise
    """
    print(f"\n🔍 Verifying data...")
    
    data_path = Path(data_dir)
    
    # Check for JSONL file
    jsonl_files = list(data_path.glob("*.jsonl"))
    if not jsonl_files:
        print("❌ No JSONL files found")
        return False
    
    jsonl_file = jsonl_files[0]
    print(f"   ✓ Found JSONL: {jsonl_file.name}")
    
    # Check JSONL content
    try:
        with open(jsonl_file, 'r') as f:
            first_line = f.readline()
            first_item = json.loads(first_line)
            num_lines = sum(1 for _ in open(jsonl_file))
        
        print(f"   ✓ Valid JSONL with {num_lines} lines")
        print(f"   ✓ Sample keys: {list(first_item.keys())}")
        
    except Exception as e:
        print(f"❌ Invalid JSONL: {e}")
        return False
    
    # Check for images directory
    images_dir = data_path / "images"
    if images_dir.exists():
        num_images = len(list(images_dir.glob("*")))
        print(f"   ✓ Found images directory with {num_images} files")
    else:
        print("⚠️  Images directory not found (may extract to different location)")
    
    print("\n✅ Data verification passed!")
    return True


def download_wmt2025_data(output_dir: str = "./data/wmt2025_raw") -> bool:
    """
    Download WMT2025 multimodal MT data.
    
    Args:
        output_dir: Directory to save data to
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 60)
    print("📥 Downloading WMT2025 Multimodal MT Data")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download text data
    text_output = output_dir / "wmt25.jsonl"
    if not download_file(URLS["text"], str(text_output), "Downloading text data (JSONL)"):
        return False
    
    # Download images
    images_zip = output_dir / "wmt25_genmt_assets.zip"
    if not download_file(URLS["images"], str(images_zip), "Downloading image assets (ZIP)"):
        return False
    
    # Extract images
    if not extract_zip(str(images_zip), str(output_dir)):
        return False
    
    # Remove zip after extraction (optional)
    print(f"\n🗑️  Cleaning up zip file...")
    try:
        images_zip.unlink()
        print(f"   ✓ Removed: {images_zip.name}")
    except Exception as e:
        print(f"   ⚠️  Could not remove zip: {e}")
    
    # Verify
    if not verify_data(str(output_dir)):
        return False
    
    print("\n" + "=" * 60)
    print("✅ Download Complete!")
    print("=" * 60)
    print(f"\n📂 Data location: {output_dir}")
    print(f"   - Text: {text_output.name}")
    print(f"   - Images: {output_dir / 'images'} (or similar)")
    
    return True


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download WMT2025 multimodal MT data")
    parser.add_argument(
        "--output-dir",
        default="./data/wmt2025_raw",
        help="Directory to save data (default: ./data/wmt2025_raw)"
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip data verification after download"
    )
    
    args = parser.parse_args()
    
    success = download_wmt2025_data(args.output_dir)
    
    if not success:
        print("\n❌ Download failed!")
        exit(1)


if __name__ == "__main__":
    main()

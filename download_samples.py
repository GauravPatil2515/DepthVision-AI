#!/usr/bin/env python3
"""
Download sample images for the DepthVision AI dashboard
"""

import requests
import os
from pathlib import Path

# Sample image URLs from COCO dataset
sample_urls = [
    ("https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800", "cat_on_table.jpg"),
    ("https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800", "living_room.jpg"),  
    ("https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800", "kitchen_scene.jpg"),
    ("https://images.unsplash.com/photo-1493809842364-78817add7ffb?w=800", "bedroom_scene.jpg"),
    ("https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=800", "office_setup.jpg")
]

def download_sample_images():
    """Download sample images to the samples directory"""
    samples_dir = Path("data/input/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading sample images...")
    
    for url, filename in sample_urls:
        output_path = samples_dir / filename
        
        if output_path.exists():
            print(f"✓ {filename} already exists, skipping...")
            continue
            
        try:
            print(f"📥 Downloading {filename}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded {filename}")
            
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
    
    print(f"\n🎉 Sample images ready in {samples_dir}")

if __name__ == "__main__":
    download_sample_images()

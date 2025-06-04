#!/usr/bin/env python3
"""
Test script for image processing with DepthVision AI
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simple_ai import DepthVisionAI
from PIL import Image

def test_image_processing():
    """Test the complete image processing pipeline"""
    print("🧪 Testing DepthVision AI Image Processing Pipeline")
    print("=" * 60)
    
    try:
        # Initialize the system
        print("🔧 Initializing DepthVision AI system...")
        system = DepthVisionAI(confidence_threshold=0.7)
        print(f"✅ System loaded: {system.system_type}")
        
        # Check for sample images
        samples_dir = Path('data/input/samples')
        if samples_dir.exists():
            sample_files = list(samples_dir.glob('*.jpg')) + list(samples_dir.glob('*.png')) + list(samples_dir.glob('*.jpeg'))
            
            if sample_files:
                print(f"📁 Found {len(sample_files)} sample images")
                
                # Test with first sample
                sample_path = sample_files[0]
                print(f"🖼️ Testing with: {sample_path.name}")
                
                # Process the image
                print("⚙️ Processing image...")
                results = system.process_image(str(sample_path), save_results=True)
                
                print("✅ Processing complete!")
                print("\n📊 Results Summary:")
                print("-" * 40)
                
                # Object detection results
                detections = results.get('object_detection', {}).get('detections', [])
                print(f"🎯 Objects detected: {len(detections)}")
                
                if detections:
                    print("📋 Detected objects:")
                    for i, detection in enumerate(detections[:5]):  # Show first 5
                        class_name = detection.get('class_name', 'unknown')
                        confidence = detection.get('confidence', 0)
                        print(f"   {i+1}. {class_name} (confidence: {confidence:.2f})")
                
                # Performance metrics
                performance = results.get('performance_metrics', {})
                processing_time = performance.get('processing_time', 0)
                detection_accuracy = performance.get('detection_accuracy', 0)
                
                print(f"⏱️ Processing time: {processing_time:.2f}s")
                print(f"🎯 Detection accuracy: {detection_accuracy:.2f}")
                
                # Depth analysis
                depth_stats = results.get('depth_estimation', {}).get('statistics', {})
                if depth_stats:
                    print(f"📏 Depth range: {depth_stats.get('min_depth', 0):.3f} - {depth_stats.get('max_depth', 1):.3f}")
                    print(f"📊 Average depth: {depth_stats.get('mean_depth', 0.5):.3f}")
                
                # 3D fusion analysis
                fusion = results.get('fusion_analysis', {})
                objects_3d = fusion.get('objects_3d', [])
                scene_structure = fusion.get('scene_structure', {})
                
                print(f"🌐 3D objects: {len(objects_3d)}")
                print(f"🏗️ Scene complexity: {scene_structure.get('scene_complexity', 0)}")
                print(f"📐 Depth layers: {scene_structure.get('depth_layers', 0)}")
                
                print("\n🎉 Image processing test completed successfully!")
                return True
                
            else:
                print("❌ No sample images found in samples directory")
                return False
        else:
            print("❌ Samples directory not found")
            
            # Try with test image
            test_image_path = Path('data/input/test_image.jpg')
            if test_image_path.exists():
                print(f"🖼️ Testing with: {test_image_path.name}")
                results = system.process_image(str(test_image_path))
                print("✅ Test image processing completed!")
                return True
            else:
                print("❌ No test images available")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_image_processing()

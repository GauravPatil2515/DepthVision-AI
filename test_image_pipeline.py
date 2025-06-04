#!/usr/bin/env python3
"""
Test script for DepthVision AI image processing pipeline
"""

import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_image_processing():
    """Test the image processing pipeline"""
    print("🧪 Testing DepthVision AI Image Processing Pipeline")
    print("=" * 60)
    
    try:
        # Import the simplified system
        from src.simple_ai import DepthVisionAI
        
        # Initialize the system
        print("🔧 Initializing DepthVision AI system...")
        system = DepthVisionAI(confidence_threshold=0.7)
        print(f"✅ System initialized successfully: {system.system_type}")
        
        # Check if sample images exist
        samples_dir = Path('data/input/samples')
        test_image_path = Path('data/input/test_image.jpg')
        
        image_to_test = None
        
        if samples_dir.exists():
            sample_files = [f for f in samples_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
            if sample_files:
                image_to_test = sample_files[0]
                print(f"📁 Found {len(sample_files)} sample images")
                print(f"🖼️  Testing with: {image_to_test.name}")
        
        if not image_to_test and test_image_path.exists():
            image_to_test = test_image_path
            print(f"🖼️  Testing with: {image_to_test.name}")
        
        if not image_to_test:
            # Create a test image
            print("📷 Creating test image...")
            test_image = Image.new('RGB', (640, 480), color='blue')
            # Add some simple shapes
            from PIL import ImageDraw
            draw = ImageDraw.Draw(test_image)
            draw.rectangle([100, 100, 300, 300], fill='red')
            draw.ellipse([400, 200, 600, 400], fill='green')
            
            os.makedirs('data/input', exist_ok=True)
            test_image.save('data/input/test_image.jpg')
            image_to_test = Path('data/input/test_image.jpg')
            print(f"✅ Created test image: {image_to_test}")
        
        # Process the image
        print(f"\n🔍 Processing image: {image_to_test}")
        results = system.process_image(str(image_to_test), save_results=True)
        
        print("\n📊 Processing Results:")
        print("-" * 40)
        
        # Object detection results
        detection_results = results.get('object_detection', {})
        detections = detection_results.get('detections', [])
        print(f"🎯 Objects detected: {len(detections)}")
        
        for i, detection in enumerate(detections[:3]):  # Show first 3
            print(f"   {i+1}. {detection['class_name']} (confidence: {detection['confidence']:.2f})")
        
        # Depth estimation results
        depth_results = results.get('depth_estimation', {})
        depth_stats = depth_results.get('statistics', {})
        print(f"🗺️  Depth map generated: {depth_stats.get('mean_depth', 'N/A'):.3f} mean depth")
        
        # Performance metrics
        performance = results.get('performance_metrics', {})
        print(f"⏱️  Processing time: {performance.get('processing_time', 0):.2f}s")
        print(f"🎯 Detection accuracy: {performance.get('detection_accuracy', 0):.1%}")
        
        # 3D fusion results
        fusion = results.get('fusion_analysis', {})
        objects_3d = fusion.get('objects_3d', [])
        scene_structure = fusion.get('scene_structure', {})
        print(f"🌐 3D objects mapped: {len(objects_3d)}")
        print(f"🏗️  Scene complexity: {scene_structure.get('scene_complexity', 0)}/10")
        
        print("\n✅ Image processing pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Image processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard_imports():
    """Test dashboard import dependencies"""
    print("\n🧪 Testing Dashboard Import Dependencies")
    print("=" * 60)
    
    required_modules = [
        'streamlit',
        'plotly',
        'PIL',
        'numpy',
        'pandas'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if not failed_imports:
        print("\n✅ All dashboard dependencies available!")
        return True
    else:
        print(f"\n❌ Missing dependencies: {failed_imports}")
        return False

if __name__ == "__main__":
    # Run tests
    deps_ok = test_dashboard_imports()
    pipeline_ok = test_image_processing()
    
    if deps_ok and pipeline_ok:
        print("\n🎉 All tests passed! Dashboard ready to run.")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")

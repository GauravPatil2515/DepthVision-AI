#!/usr/bin/env python3
"""
Test script for DepthVision AI Dashboard
Verifies all components are working correctly
"""

import sys
import os
import importlib
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing module imports...")
    
    modules_to_test = [
        'streamlit',
        'plotly',
        'cv2',
        'PIL',
        'numpy',
        'pandas',
        'requests'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_src_modules():
    """Test if our source modules can be imported"""
    print("\n🔧 Testing source modules...")
    
    src_modules = [
        'src.models.fusion_model',
        'src.models.object_detection', 
        'src.models.depth_estimation',
        'src.utils.visualization',
        'src.utils.metrics',
        'src.config.config'
    ]
    
    failed_imports = []
    
    for module in src_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_sample_images():
    """Test if sample images are available"""
    print("\n🖼️ Testing sample images...")
    
    samples_dir = Path("data/input/samples")
    
    if not samples_dir.exists():
        print("❌ Samples directory doesn't exist")
        return False
    
    image_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
    
    if len(image_files) == 0:
        print("❌ No sample images found")
        return False
    
    print(f"✅ Found {len(image_files)} sample images:")
    for img in image_files:
        print(f"   📸 {img.name}")
    
    return True

def test_model_loading():
    """Test if AI models can be loaded"""
    print("\n🤖 Testing model loading...")
    
    try:
        from src.models.fusion_model import DepthVisionAI
        
        print("📥 Loading DepthVision AI system...")
        system = DepthVisionAI()
        
        # Test system info
        info = system.get_system_info()
        print("✅ System loaded successfully")
        print(f"   🎯 Detection backend: {info['detection_backend']}")
        print(f"   📏 Depth backend: {info['depth_backend']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_image_processing():
    """Test image processing with a sample"""
    print("\n🔄 Testing image processing...")
    
    try:
        from src.models.fusion_model import DepthVisionAI
        
        # Find a sample image
        samples_dir = Path("data/input/samples")
        sample_images = list(samples_dir.glob("*.jpg"))
        
        if not sample_images:
            print("❌ No sample images for testing")
            return False
        
        test_image = str(sample_images[0])
        print(f"📸 Processing: {sample_images[0].name}")
        
        system = DepthVisionAI()
        results = system.process_image(test_image, save_results=False)
        
        # Check results structure
        required_keys = ['object_detection', 'depth_estimation', 'fusion_analysis']
        
        for key in required_keys:
            if key not in results:
                print(f"❌ Missing result key: {key}")
                return False
        
        # Check detection results
        detections = results['object_detection']['detections']
        print(f"✅ Detected {len(detections)} objects")
        
        # Check depth results
        if 'depth_map' in results['depth_estimation']:
            print("✅ Depth map generated")
        
        # Check fusion results
        objects_3d = results['fusion_analysis']['objects_3d']
        print(f"✅ Created {len(objects_3d)} 3D objects")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 DepthVision AI Dashboard - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Source Modules", test_src_modules),
        ("Sample Images", test_sample_images),
        ("Model Loading", test_model_loading),
        ("Image Processing", test_image_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard is ready to use.")
        print("\n🚀 To start the dashboard, run:")
        print("   streamlit run streamlit_app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

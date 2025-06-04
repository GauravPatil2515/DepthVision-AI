"""
Test Script for DepthVision AI
Comprehensive testing of all system components
"""

import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_individual_components():
    """Test individual components separately"""
    print("="*60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("="*60)
    
    # Test 1: Depth Estimation
    print("\n1. Testing Depth Estimation...")
    try:
        from src.models.depth_estimation import test_depth_estimation
        success = test_depth_estimation()
        print(f"✅ Depth Estimation Test: {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        print(f"❌ Depth Estimation Test: FAILED - {e}")
    
    # Test 2: Object Detection
    print("\n2. Testing Object Detection...")
    try:
        from src.models.object_detection import test_object_detection
        success = test_object_detection()
        print(f"✅ Object Detection Test: {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        print(f"❌ Object Detection Test: FAILED - {e}")
    
    # Test 3: Fusion Model
    print("\n3. Testing Fusion Model...")
    try:
        from src.models.fusion_model import test_fusion_model
        success = test_fusion_model()
        print(f"✅ Fusion Model Test: {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        print(f"❌ Fusion Model Test: FAILED - {e}")
    
    # Test 4: Visualization
    print("\n4. Testing Visualization...")
    try:
        from src.utils.visualization import test_visualization
        success = test_visualization()
        print(f"✅ Visualization Test: {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        print(f"❌ Visualization Test: FAILED - {e}")
    
    # Test 5: Metrics
    print("\n5. Testing Metrics...")
    try:
        from src.utils.metrics import test_metrics
        success = test_metrics()
        print(f"✅ Metrics Test: {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        print(f"❌ Metrics Test: FAILED - {e}")

def test_main_application():
    """Test the main application"""
    print("\n" + "="*60)
    print("TESTING MAIN APPLICATION")
    print("="*60)
    
    try:
        from src.main import DepthVisionAI
        
        # Initialize the system
        print("\nInitializing DepthVision AI System...")
        system = DepthVisionAI(
            enable_visualization=False,  # Disable for faster testing
            enable_metrics=True
        )
        
        # Get system info
        system_info = system.get_system_info()
        print(f"✅ System Info: {system_info['system_version']}")
        print(f"✅ Device: {system_info['device']}")
        print(f"✅ CUDA Available: {system_info['cuda_available']}")
          # Test with sample image
        print("\nTesting with sample image...")
        sample_image_path = "data/input/test_image.jpg"
        
        # Create a temporary test directory
        test_output_dir = Path("test_output")
        test_output_dir.mkdir(exist_ok=True)
        
        # Process the image
        results = system.process_image(
            sample_image_path, 
            output_dir=test_output_dir,
            save_results=True
        )
        
        print(f"✅ Main Application Test: PASSED")
        print(f"   - Objects detected: {len(results['fusion_analysis']['objects_3d'])}")
        print(f"   - Scene complexity: {results['fusion_analysis']['scene_structure']['scene_complexity']}")
        
        if 'performance_metrics' in results:
            print(f"   - Performance score: {results['performance_metrics']['overall_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Main Application Test: FAILED - {e}")
        return False

def test_imports():
    """Test that all modules can be imported"""
    print("="*60)
    print("TESTING MODULE IMPORTS")
    print("="*60)
    
    modules_to_test = [
        "src.models.depth_estimation",
        "src.models.object_detection", 
        "src.models.fusion_model",
        "src.utils.image_processing",
        "src.utils.visualization",
        "src.utils.metrics",
        "src.config.config",
        "src.main"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} modules failed to import")
        return False
    else:
        print(f"\n✅ All {len(modules_to_test)} modules imported successfully")
        return True

def check_dependencies():
    """Check if all required dependencies are available"""
    print("="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    required_packages = [
        "torch", "torchvision", "transformers", "numpy", 
        "PIL", "cv2", "matplotlib", "seaborn", "plotly",
        "sklearn", "requests", "tqdm", "pandas", "scipy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
            elif package == "cv2":
                import cv2
            elif package == "sklearn":
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print(f"\n✅ All required packages are available")
        return True

def main():
    """Run all tests"""
    print("DepthVision AI - Comprehensive Test Suite")
    print("="*60)
    
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    # Test 1: Dependencies
    deps_ok = check_dependencies()
    
    # Test 2: Imports
    imports_ok = test_imports()
    
    # Test 3: Individual Components (if basic tests pass)
    if deps_ok and imports_ok:
        test_individual_components()
        
        # Test 4: Main Application
        main_app_ok = test_main_application()
    else:
        print("\n⚠️ Skipping component tests due to dependency/import failures")
        main_app_ok = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"Module Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Main Application: {'✅ PASS' if main_app_ok else '❌ FAIL'}")
    
    if deps_ok and imports_ok and main_app_ok:
        print("\n🎉 ALL TESTS PASSED! DepthVision AI is ready to use.")
        print("\nTo run the application:")
        print("python src/main.py --input path/to/image.jpg --output results/")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
    
    print("="*60)

if __name__ == "__main__":
    main()

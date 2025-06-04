#!/usr/bin/env python3
"""
Test script for the simplified AI system
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.simple_ai import DepthVisionAI

def test_system():
    """Test the simplified AI system with a sample image"""
    print("🧪 Testing DepthVision AI System with sample image")
    
    # Initialize system
    system = DepthVisionAI()
    print(f"✅ System initialized: {system.system_type}")
    
    # Test with sample image
    sample_image = "data/input/samples/living_room.jpg"
    print(f"📸 Processing image: {sample_image}")
    
    try:
        results = system.process_image(sample_image)
        
        print("\n📊 Processing Results:")
        print(f"  - Objects detected: {results['performance_metrics']['objects_detected']}")
        print(f"  - Processing time: {results['performance_metrics']['processing_time']:.2f}s")
        print(f"  - Detection accuracy: {results['performance_metrics']['detection_accuracy']:.2f}")
        print(f"  - Scene complexity: {results['performance_metrics']['scene_complexity']}")
        
        print("\n🎯 Object Detections:")
        for i, obj in enumerate(results['object_detection']['detections']):
            print(f"  {i+1}. {obj['class_name']} (confidence: {obj['confidence']:.2f})")
        
        print("\n🌊 Depth Analysis:")
        depth_stats = results['depth_estimation']['statistics']
        print(f"  - Depth range: {depth_stats['min_depth']:.3f} - {depth_stats['max_depth']:.3f}")
        print(f"  - Mean depth: {depth_stats['mean_depth']:.3f}")
        
        print("\n🔮 3D Fusion:")
        print(f"  - Objects in 3D: {len(results['fusion_analysis']['objects_3d'])}")
        scene_struct = results['fusion_analysis']['scene_structure']
        print(f"  - Scene complexity: {scene_struct['scene_complexity']}")
        print(f"  - Depth layers: {scene_struct['depth_layers']}")
        
        print("\n✅ All tests passed! System is ready for dashboard.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)

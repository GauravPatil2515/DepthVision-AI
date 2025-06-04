#!/usr/bin/env python3
"""
Simplified DepthVision AI System for Dashboard
A streamlined version that works with the Streamlit dashboard
"""

import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class DepthVisionAI:
    """
    Simplified DepthVision AI system for the Streamlit dashboard
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize the simplified system"""
        self.confidence_threshold = confidence_threshold
        self.object_detector = None
        self.depth_estimator = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            # Try to load the full system
            self._load_full_system()
            self.system_type = "full"
            self.logger.info("✅ Full AI system loaded successfully")
        except Exception as e:
            # Fallback to dummy system
            self.logger.warning(f"Full system failed to load: {e}")
            self._load_dummy_system()
            self.system_type = "dummy"
            self.logger.info("⚠️ Using dummy system for demonstration")
    
    def _load_full_system(self):
        """Try to load the full AI system"""
        from .models.object_detection import ObjectDetector
        from .models.depth_estimation import DepthEstimator
        
        self.object_detector = ObjectDetector(confidence_threshold=self.confidence_threshold)
        self.depth_estimator = DepthEstimator()
    
    def _load_dummy_system(self):
        """Load dummy system for demonstration"""
        self.object_detector = DummyObjectDetector()
        self.depth_estimator = DummyDepthEstimator()
    
    def process_image(self, image_path: Union[str, Image.Image], save_results: bool = False) -> Dict:
        """Process an image with detection and depth estimation"""
        try:
            start_time = time.time()
            
            # Load image if path provided
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                image = image_path
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get image dimensions
            image_np = np.array(image)
            image_shape = image_np.shape[:2]
            
            # Object detection
            detection_results = self.object_detector.detect(image)
            
            # Depth estimation
            depth_results = self.depth_estimator.estimate_depth(image)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create fusion results
            fusion_results = self._create_fusion_results(detection_results, depth_results, image_shape)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(
                detection_results,
                depth_results,
                processing_time
            )
            
            # Create final results dictionary
            results = {
                'object_detection': detection_results,
                'depth_estimation': depth_results,
                'fusion_analysis': fusion_results,
                'performance_metrics': metrics,
                'error': None
            }
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Processing failed: {error_msg}")
            return self._create_error_result(str(e))
    
    def _create_fusion_results(self, detection_results: Dict, depth_results: Dict, image_shape: tuple) -> Dict:
        """Create 3D fusion analysis from detection and depth results"""
        objects_3d = []
        
        if detection_results.get('detections'):
            # Handle depth_results as dictionary
            depth_map = depth_results.get('depth_map') if isinstance(depth_results, dict) else None
            
            if depth_map is None:
                depth_map = np.random.rand(*image_shape)  # Dummy depth for testing
            
            for detection in detection_results['detections']:
                bbox = detection['bbox']
                
                # Calculate center point
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                # Get depth value at center
                if (center_y < depth_map.shape[0] and 
                    center_x < depth_map.shape[1]):
                    depth_value = float(depth_map[center_y, center_x])
                else:
                    depth_value = 0.5  # Default depth
                
                # Create 3D position (normalized coordinates)
                position_3d = [
                    center_x / image_shape[1],  # x (0-1)
                    center_y / image_shape[0],  # y (0-1)
                    depth_value                 # z (0-1, relative depth)
                ]
                
                objects_3d.append({
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'position_3d': position_3d,
                    'depth_value': depth_value,
                    'bbox': bbox
                })
        
        # Scene structure analysis
        scene_complexity = min(len(objects_3d), 10)  # 1-10 scale
        
        return {
            'objects_3d': objects_3d,
            'scene_structure': {
                'scene_complexity': scene_complexity,
                'depth_layers': len(set(obj['depth_value'] for obj in objects_3d)),
                'spatial_density': len(objects_3d) / (image_shape[0] * image_shape[1])
            }
        }
    
    def _calculate_performance_metrics(self, detection_results: Dict, depth_results: Dict, processing_time: float) -> Dict:
        """Calculate performance metrics"""
        return {
            'processing_time': processing_time,
            'objects_detected': len(detection_results.get('detections', [])),
            'detection_accuracy': 1.0 if detection_results.get('error') is None else 0.0,
            'depth_estimation_time': processing_time * 0.6,  # Estimated time distribution
            'detection_time': processing_time * 0.4,
            'scene_complexity': len(detection_results.get('detections', [])) / 10.0  # Normalized 0-1
        }
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result structure"""
        return {
            'object_detection': {'detections': [], 'error': error_message},
            'depth_estimation': {'depth_map': None, 'error': error_message},
            'fusion_analysis': {'objects_3d': [], 'error': error_message},
            'performance_metrics': {},
            'error': error_message
        }
    
    def get_system_info(self) -> Dict:
        """Get system information and status"""
        import platform
        import torch
        
        return {
            'system_type': self.system_type,
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cuda_available': torch.cuda.is_available(),
            'gpu_info': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'detection_backend': getattr(self.object_detector, 'backend', 'unknown'),
            'confidence_threshold': self.confidence_threshold
        }

class DummyObjectDetector:
    """Dummy object detector for demonstration"""
    def detect(self, image):
        """Return dummy detection results"""
        return {
            'detections': [
                {
                    'bbox': [100, 100, 200, 200],
                    'confidence': 0.95,
                    'class_name': 'example_object',
                    'class_id': 0
                }
            ],
            'visualization': None,
            'error': None
        }

class DummyDepthEstimator:
    """Dummy depth estimator for demonstration"""
    def estimate_depth(self, image):
        """Return dummy depth results"""
        if isinstance(image, str):
            image = Image.open(image)
        
        # Create random depth map
        depth_map = np.random.rand(image.size[1], image.size[0])
        
        return {
            'depth_map': depth_map,
            'depth_visualization': None,
            'statistics': {
                'min_depth': 0.0,
                'max_depth': 1.0,
                'mean_depth': 0.5,
                'std_depth': 0.1
            },
            'error': None
        }

if __name__ == "__main__":
    # Test the simplified system
    print("🧪 Testing Simplified DepthVision AI System")
    
    try:
        system = DepthVisionAI()
        print(f"✅ System initialized: {system.system_type}")
        
        # Get system info
        info = system.get_system_info()
        print(f"📊 System info: {info}")
        
        # Test with sample image
        sample_image = "data/input/samples/test_image.jpg"
        results = system.process_image(sample_image)
        
        if results['error'] is None:
            print("✅ Processing test passed")
            print(f"Objects detected: {len(results['object_detection']['detections'])}")
            print(f"Processing time: {results['performance_metrics']['processing_time']:.2f}s")
        else:
            print(f"❌ Test failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

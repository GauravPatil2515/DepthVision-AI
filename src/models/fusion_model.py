"""
Multi-modal Fusion Module
Combines object detection and depth estimation for comprehensive 3D scene understanding
"""

import numpy as np
from PIL import Image
import torch
import cv2
from typing import Dict, List, Tuple, Union
import logging
from .depth_estimation import DepthEstimator
from .object_detection import ObjectDetector
from ..utils.image_processing import ImageProcessor

class FusionModel:
    """
    Multi-modal fusion system for comprehensive 3D scene understanding
    Combines Detectron2 object detection with Intel MiDaS depth estimation
    """
    
    def __init__(self, depth_model_name: str = "Intel/dpt-large", 
                 detection_confidence: float = 0.5):
        """
        Initialize the fusion model
        
        Args:
            depth_model_name: Depth estimation model identifier
            detection_confidence: Object detection confidence threshold
        """
        logging.info("Initializing FusionModel...")
        
        # Initialize individual models
        self.depth_estimator = DepthEstimator(model_name=depth_model_name)
        self.object_detector = ObjectDetector(confidence_threshold=detection_confidence)
        self.image_processor = ImageProcessor()
        
        logging.info("FusionModel initialized successfully")
    
    def analyze_scene(self, image: Union[Image.Image, str, np.ndarray]) -> Dict:
        """
        Perform comprehensive 3D scene analysis
        
        Args:
            image: Input image
            
        Returns:
            Complete scene analysis results
        """
        logging.info("Starting comprehensive scene analysis...")
        
        # Load and prepare image
        pil_image = self.image_processor.load_image(image)
        
        # Perform depth estimation
        logging.info("Performing depth estimation...")
        depth_array, depth_visualization = self.depth_estimator.estimate_depth(pil_image)
        depth_stats = self.depth_estimator.get_depth_statistics(depth_array)
        
        # Perform object detection
        logging.info("Performing object detection...")
        detection_results = self.object_detector.detect_objects(pil_image)
        detection_stats = self.object_detector.get_detection_statistics(detection_results)
        
        # Perform fusion analysis
        logging.info("Performing multi-modal fusion...")
        fusion_results = self._fuse_depth_and_detection(
            depth_array, detection_results, pil_image
        )
        
        # Compile comprehensive results
        scene_analysis = {
            'image_info': {
                'width': pil_image.width,
                'height': pil_image.height,
                'channels': len(np.array(pil_image).shape)
            },
            'depth_estimation': {
                'depth_array': depth_array,
                'depth_visualization': depth_visualization,
                'statistics': depth_stats
            },
            'object_detection': {
                'results': detection_results,
                'statistics': detection_stats
            },
            'fusion_analysis': fusion_results,
            'spatial_analysis': self._calculate_spatial_metrics(fusion_results)        }
        
        logging.info("Scene analysis completed successfully")
        return scene_analysis
    
    def _fuse_depth_and_detection(self, depth_array: np.ndarray, 
                                 detection_results: Dict, 
                                 original_image: Image.Image) -> Dict:
        """
        Fuse depth estimation with object detection results
        
        Args:
            depth_array: Depth estimation array
            detection_results: Object detection results
            original_image: Original input image
            
        Returns:
            Fusion analysis results
        """
        fusion_results = {
            'objects_3d': [],
            'depth_segmented_objects': [],
            'spatial_relationships': [],
            'scene_structure': {}
        }
        
        if not detection_results['boxes']:
            logging.warning("No objects detected, returning empty fusion results")
            # Still analyze scene structure even without objects
            fusion_results['scene_structure'] = self._analyze_scene_structure(
                depth_array, []
            )
            return fusion_results
        
        # Process each detected object
        for i, (box, class_name, score) in enumerate(zip(
            detection_results['boxes'],
            detection_results['class_names'], 
            detection_results['scores']
        )):
            # Extract object region from depth map
            x1, y1, x2, y2 = map(int, box)
            object_depth_region = depth_array[y1:y2, x1:x2]
            
            # Calculate 3D object properties
            object_3d = self._calculate_3d_properties(
                object_depth_region, box, class_name, score, i
            )
            
            fusion_results['objects_3d'].append(object_3d)
            
            # Create depth-segmented object visualization
            segmented_object = self._create_depth_segmented_object(
                original_image, depth_array, box, object_depth_region
            )
            fusion_results['depth_segmented_objects'].append(segmented_object)
        
        # Calculate spatial relationships between objects
        fusion_results['spatial_relationships'] = self._calculate_spatial_relationships(
            fusion_results['objects_3d']
        )
        
        # Analyze overall scene structure
        fusion_results['scene_structure'] = self._analyze_scene_structure(
            depth_array, fusion_results['objects_3d']
        )
        
        return fusion_results
    
    def _calculate_3d_properties(self, object_depth_region: np.ndarray, 
                                box: List[float], class_name: str, 
                                score: float, object_id: int) -> Dict:
        """
        Calculate 3D properties for a detected object
        
        Args:
            object_depth_region: Depth values within object bounding box
            box: Bounding box coordinates [x1, y1, x2, y2]
            class_name: Object class name
            score: Detection confidence score
            object_id: Unique object identifier
            
        Returns:
            3D object properties
        """
        x1, y1, x2, y2 = box
        width_2d = x2 - x1
        height_2d = y2 - y1
        
        # Calculate depth statistics for the object
        if object_depth_region.size > 0:
            avg_depth = float(np.mean(object_depth_region))
            min_depth = float(np.min(object_depth_region))
            max_depth = float(np.max(object_depth_region))
            depth_variance = float(np.var(object_depth_region))
        else:
            avg_depth = min_depth = max_depth = depth_variance = 0.0
        
        # Estimate 3D volume (simplified calculation)
        depth_range = max_depth - min_depth
        estimated_volume = width_2d * height_2d * depth_range
        
        return {
            'object_id': object_id,
            'class_name': class_name,
            'confidence': score,
            'bounding_box_2d': box,
            'center_2d': [(x1 + x2) / 2, (y1 + y2) / 2],
            'dimensions_2d': [width_2d, height_2d],
            'depth_properties': {
                'average_depth': avg_depth,
                'min_depth': min_depth,
                'max_depth': max_depth,
                'depth_variance': depth_variance,
                'depth_range': depth_range
            },
            'estimated_3d_properties': {
                'estimated_volume': estimated_volume,
                'depth_consistency': 1.0 - (depth_variance / (avg_depth + 1e-6))
            },
            'spatial_position': self._classify_spatial_position(avg_depth)
        }
    
    def _classify_spatial_position(self, depth_value: float) -> str:
        """
        Classify spatial position based on depth value
        
        Args:
            depth_value: Object's average depth
            
        Returns:
            Spatial position classification
        """
        # These thresholds can be adjusted based on your specific use case
        if depth_value < 0.3:
            return "foreground"
        elif depth_value < 0.7:
            return "middle_ground"
        else:
            return "background"
    
    def _create_depth_segmented_object(self, original_image: Image.Image,
                                     depth_array: np.ndarray,
                                     box: List[float],
                                     object_depth_region: np.ndarray) -> Dict:
        """
        Create depth-segmented visualization of an object
        
        Args:
            original_image: Original input image
            depth_array: Full depth array
            box: Object bounding box
            object_depth_region: Depth values for the object
            
        Returns:
            Segmented object data
        """
        x1, y1, x2, y2 = map(int, box)
        
        # Extract object region from original image
        img_array = np.array(original_image)
        object_image_region = img_array[y1:y2, x1:x2]
        
        # Create depth-colored version
        depth_colored = self.image_processor.convert_depth_to_colormap(
            object_depth_region, colormap='plasma'
        )
        
        return {
            'bounding_box': box,
            'original_region': object_image_region,
            'depth_region': object_depth_region,
            'depth_colored': depth_colored
        }
    
    def _calculate_spatial_relationships(self, objects_3d: List[Dict]) -> List[Dict]:
        """
        Calculate spatial relationships between detected objects
        
        Args:
            objects_3d: List of 3D object properties
            
        Returns:
            List of spatial relationship descriptions
        """
        relationships = []
        
        for i, obj1 in enumerate(objects_3d):
            for j, obj2 in enumerate(objects_3d[i+1:], i+1):
                # Calculate 2D distance
                center1 = obj1['center_2d']
                center2 = obj2['center_2d']
                distance_2d = np.sqrt((center1[0] - center2[0])**2 + 
                                    (center1[1] - center2[1])**2)
                
                # Calculate depth difference
                depth_diff = abs(obj1['depth_properties']['average_depth'] - 
                               obj2['depth_properties']['average_depth'])
                
                # Determine relationship type
                relationship_type = self._determine_relationship_type(
                    obj1, obj2, distance_2d, depth_diff
                )
                
                relationships.append({
                    'object1_id': obj1['object_id'],
                    'object1_class': obj1['class_name'],
                    'object2_id': obj2['object_id'],
                    'object2_class': obj2['class_name'],
                    'distance_2d': distance_2d,
                    'depth_difference': depth_diff,
                    'relationship_type': relationship_type
                })
        
        return relationships
    
    def _determine_relationship_type(self, obj1: Dict, obj2: Dict, 
                                   distance_2d: float, depth_diff: float) -> str:
        """
        Determine the type of spatial relationship between two objects
        
        Args:
            obj1: First object properties
            obj2: Second object properties
            distance_2d: 2D distance between objects
            depth_diff: Depth difference between objects
            
        Returns:
            Relationship type description
        """
        # Thresholds (can be adjusted based on requirements)
        close_distance_threshold = 100  # pixels
        significant_depth_threshold = 0.1
        
        if distance_2d < close_distance_threshold:
            if depth_diff < significant_depth_threshold:
                return "adjacent_same_depth"
            else:
                return "overlapping_different_depth"
        else:
            if depth_diff < significant_depth_threshold:
                return "distant_same_depth"
            else:
                depth1 = obj1['depth_properties']['average_depth']
                depth2 = obj2['depth_properties']['average_depth']
                if depth1 < depth2:
                    return f"{obj1['class_name']}_in_front_of_{obj2['class_name']}"
                else:
                    return f"{obj2['class_name']}_in_front_of_{obj1['class_name']}"
    
    def _analyze_scene_structure(self, depth_array: np.ndarray, 
                               objects_3d: List[Dict]) -> Dict:
        """
        Analyze overall scene structure
        
        Args:
            depth_array: Full scene depth array
            objects_3d: List of 3D object properties
            
        Returns:
            Scene structure analysis
        """
        # Calculate global scene statistics
        scene_depth_stats = {
            'mean_depth': float(np.mean(depth_array)),
            'std_depth': float(np.std(depth_array)),
            'min_depth': float(np.min(depth_array)),
            'max_depth': float(np.max(depth_array))
        }
        
        # Analyze depth distribution
        depth_hist, depth_bins = np.histogram(depth_array.flatten(), bins=50)
        
        # Count objects by spatial position
        position_counts = {}
        for obj in objects_3d:
            pos = obj['spatial_position']
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        return {
            'scene_depth_statistics': scene_depth_stats,
            'depth_distribution': {
                'histogram': depth_hist.tolist(),
                'bins': depth_bins.tolist()
            },
            'object_spatial_distribution': position_counts,
            'scene_complexity': len(objects_3d),
            'depth_layers': self._identify_depth_layers(objects_3d)
        }
    
    def _identify_depth_layers(self, objects_3d: List[Dict]) -> List[Dict]:
        """
        Identify distinct depth layers in the scene
        
        Args:
            objects_3d: List of 3D object properties
            
        Returns:
            List of depth layer descriptions
        """
        if not objects_3d:
            return []
        
        # Extract depths and sort
        depths = [obj['depth_properties']['average_depth'] for obj in objects_3d]
        sorted_depths = sorted(set(depths))
        
        # Group objects into layers (simplified clustering)
        layers = []
        current_layer = []
        current_depth = sorted_depths[0]
        depth_threshold = 0.1  # Threshold for grouping into same layer
        
        for depth in sorted_depths:
            if abs(depth - current_depth) <= depth_threshold:
                current_layer.append(depth)
            else:
                if current_layer:
                    layers.append({
                        'layer_id': len(layers),
                        'average_depth': np.mean(current_layer),
                        'depth_range': [min(current_layer), max(current_layer)],
                        'object_count': len(current_layer)
                    })
                current_layer = [depth]
                current_depth = depth
        
        # Add the last layer
        if current_layer:
            layers.append({
                'layer_id': len(layers),
                'average_depth': np.mean(current_layer),
                'depth_range': [min(current_layer), max(current_layer)],
                'object_count': len(current_layer)
            })
        
        return layers
    
    def _calculate_spatial_metrics(self, fusion_results: Dict) -> Dict:
        """
        Calculate advanced spatial metrics for the scene
        
        Args:
            fusion_results: Fusion analysis results
            
        Returns:
            Spatial metrics dictionary
        """
        objects_3d = fusion_results['objects_3d']
        relationships = fusion_results['spatial_relationships']
        
        if not objects_3d:
            return {'total_objects': 0}
        
        # Calculate scene density
        total_area = sum(obj['dimensions_2d'][0] * obj['dimensions_2d'][1] 
                        for obj in objects_3d)
        
        # Calculate depth diversity
        depths = [obj['depth_properties']['average_depth'] for obj in objects_3d]
        depth_diversity = np.std(depths) if len(depths) > 1 else 0
        
        # Calculate relationship density
        relationship_density = len(relationships) / len(objects_3d) if objects_3d else 0
        
        return {
            'total_objects': len(objects_3d),
            'scene_density': total_area,
            'depth_diversity': depth_diversity,
            'relationship_density': relationship_density,
            'unique_classes': len(set(obj['class_name'] for obj in objects_3d)),
            'average_confidence': np.mean([obj['confidence'] for obj in objects_3d])
        }

def test_fusion_model():
    """Test function to verify fusion model functionality"""
    
    # Initialize the fusion model
    fusion_model = FusionModel()
    
    # Test with a sample image (create a simple test image)
    import numpy as np
    from PIL import Image
    import os
    
    # Create test directory if it doesn't exist
    os.makedirs("data/input", exist_ok=True)
    
    # Create or use existing test image
    test_image_path = "data/input/test_image.jpg"
    if not os.path.exists(test_image_path):
        # Create a simple test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        Image.fromarray(test_img).save(test_image_path)
    
    try:
        # Perform comprehensive scene analysis
        scene_analysis = fusion_model.analyze_scene(test_image_path)
        
        print("Fusion model test completed successfully!")
        print(f"Detected {scene_analysis['spatial_analysis']['total_objects']} objects")
        print(f"Scene complexity: {scene_analysis['fusion_analysis']['scene_structure']['scene_complexity']}")
        print(f"Depth layers: {len(scene_analysis['fusion_analysis']['scene_structure']['depth_layers'])}")
        
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_fusion_model()

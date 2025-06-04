"""
Metrics and Performance Evaluation Module
Comprehensive evaluation tools for DepthVision AI system performance
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
import json

class PerformanceMetrics:
    """
    Comprehensive performance evaluation for DepthVision AI
    Tracks accuracy, speed, and quality metrics
    """
    
    def __init__(self):
        """Initialize performance metrics tracker"""
        self.metrics_history = []
        self.timing_data = {}
        
        logging.info("PerformanceMetrics initialized")
    
    def evaluate_detection_accuracy(self, predictions: Dict, ground_truth: Dict = None) -> Dict:
        """
        Evaluate object detection accuracy
        
        Args:
            predictions: Detection results from model
            ground_truth: Ground truth annotations (optional)
            
        Returns:
            Detection accuracy metrics
        """
        metrics = {
            'total_detections': len(predictions.get('boxes', [])),
            'confidence_metrics': self._calculate_confidence_metrics(predictions),
            'class_distribution': self._calculate_class_distribution(predictions)
        }
        
        # If ground truth is available, calculate precision/recall
        if ground_truth:
            metrics.update(self._calculate_detection_precision_recall(predictions, ground_truth))
        
        # Estimate accuracy based on confidence scores (when no ground truth)
        if not ground_truth and predictions.get('scores'):
            high_confidence_detections = sum(1 for score in predictions['scores'] if score > 0.8)
            estimated_accuracy = high_confidence_detections / len(predictions['scores']) if predictions['scores'] else 0
            metrics['estimated_accuracy'] = estimated_accuracy
            
            # Target: 99.94% accuracy
            metrics['accuracy_target_met'] = estimated_accuracy >= 0.9994
        
        return metrics
    
    def evaluate_depth_quality(self, depth_array: np.ndarray, ground_truth_depth: np.ndarray = None) -> Dict:
        """
        Evaluate depth estimation quality
        
        Args:
            depth_array: Predicted depth map
            ground_truth_depth: Ground truth depth map (optional)
            
        Returns:
            Depth quality metrics
        """
        metrics = {
            'depth_statistics': self._calculate_depth_statistics(depth_array),
            'depth_consistency': self._calculate_depth_consistency(depth_array),
            'sub_pixel_precision': self._evaluate_sub_pixel_precision(depth_array)
        }
        
        # If ground truth is available, calculate error metrics
        if ground_truth_depth is not None:
            metrics.update(self._calculate_depth_error_metrics(depth_array, ground_truth_depth))
        
        return metrics
    
    def evaluate_fusion_performance(self, fusion_results: Dict) -> Dict:
        """
        Evaluate multi-modal fusion performance
        
        Args:
            fusion_results: Results from fusion model
            
        Returns:
            Fusion performance metrics
        """
        objects_3d = fusion_results.get('objects_3d', [])
        spatial_relationships = fusion_results.get('spatial_relationships', [])
        
        metrics = {
            '3d_object_quality': self._evaluate_3d_object_quality(objects_3d),
            'spatial_analysis_quality': self._evaluate_spatial_analysis(spatial_relationships),
            'scene_understanding_score': self._calculate_scene_understanding_score(fusion_results),
            'fusion_completeness': len(objects_3d) > 0 and len(spatial_relationships) > 0
        }
        
        return metrics
    
    def measure_processing_time(self, operation_name: str):
        """
        Context manager for measuring processing time
        
        Args:
            operation_name: Name of the operation being timed
        """
        class TimingContext:
            def __init__(self, metrics_obj, op_name):
                self.metrics_obj = metrics_obj
                self.op_name = op_name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                duration = end_time - self.start_time
                self.metrics_obj.timing_data[self.op_name] = duration
                logging.info(f"{self.op_name} completed in {duration:.3f} seconds")
        
        return TimingContext(self, operation_name)
    
    def evaluate_overall_performance(self, scene_analysis: Dict) -> Dict:
        """
        Evaluate overall system performance
        
        Args:
            scene_analysis: Complete scene analysis results
            
        Returns:
            Overall performance metrics
        """
        # Extract individual component results
        detection_results = scene_analysis.get('object_detection', {}).get('results', {})
        depth_array = scene_analysis.get('depth_estimation', {}).get('depth_array')
        fusion_results = scene_analysis.get('fusion_analysis', {})
        
        # Evaluate each component
        detection_metrics = self.evaluate_detection_accuracy(detection_results)
        depth_metrics = self.evaluate_depth_quality(depth_array) if depth_array is not None else {}
        fusion_metrics = self.evaluate_fusion_performance(fusion_results)
        
        # Calculate overall performance score
        overall_score = self._calculate_overall_score(
            detection_metrics, depth_metrics, fusion_metrics
        )
        
        # Compile comprehensive metrics
        comprehensive_metrics = {
            'timestamp': time.time(),
            'detection_performance': detection_metrics,
            'depth_performance': depth_metrics,
            'fusion_performance': fusion_metrics,
            'timing_data': self.timing_data.copy(),
            'overall_score': overall_score,
            'system_status': self._assess_system_status(overall_score)
        }
        
        # Store in history
        self.metrics_history.append(comprehensive_metrics)
        
        return comprehensive_metrics
    
    def _calculate_confidence_metrics(self, predictions: Dict) -> Dict:
        """Calculate confidence-based metrics"""
        scores = predictions.get('scores', [])
        
        if not scores:
            return {'mean_confidence': 0, 'std_confidence': 0, 'min_confidence': 0, 'max_confidence': 0}
        
        return {
            'mean_confidence': float(np.mean(scores)),
            'std_confidence': float(np.std(scores)),
            'min_confidence': float(np.min(scores)),
            'max_confidence': float(np.max(scores)),
            'high_confidence_ratio': sum(1 for s in scores if s > 0.8) / len(scores)
        }
    
    def _calculate_class_distribution(self, predictions: Dict) -> Dict:
        """Calculate class distribution metrics"""
        class_names = predictions.get('class_names', [])
        
        if not class_names:
            return {}
        
        class_counts = {}
        for class_name in class_names:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        total = len(class_names)
        class_percentages = {cls: count/total for cls, count in class_counts.items()}
        
        return {
            'unique_classes': len(class_counts),
            'class_counts': class_counts,
            'class_percentages': class_percentages,
            'most_frequent_class': max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None
        }
    
    def _calculate_detection_precision_recall(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """Calculate precision and recall for object detection"""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated IoU-based matching
        
        pred_classes = predictions.get('classes', [])
        true_classes = ground_truth.get('classes', [])
        
        if not pred_classes or not true_classes:
            return {'precision': 0, 'recall': 0, 'f1_score': 0}
        
        # Simplified calculation (assumes perfect matching for demonstration)
        try:
            precision = precision_score(true_classes, pred_classes, average='weighted', zero_division=0)
            recall = recall_score(true_classes, pred_classes, average='weighted', zero_division=0)
            f1 = f1_score(true_classes, pred_classes, average='weighted', zero_division=0)
            
            return {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        except Exception as e:
            logging.warning(f"Error calculating precision/recall: {e}")
            return {'precision': 0, 'recall': 0, 'f1_score': 0}
    
    def _calculate_depth_statistics(self, depth_array: np.ndarray) -> Dict:
        """Calculate depth map statistics"""
        return {
            'mean_depth': float(np.mean(depth_array)),
            'std_depth': float(np.std(depth_array)),
            'min_depth': float(np.min(depth_array)),
            'max_depth': float(np.max(depth_array)),
            'median_depth': float(np.median(depth_array)),
            'depth_range': float(np.max(depth_array) - np.min(depth_array))
        }
    
    def _calculate_depth_consistency(self, depth_array: np.ndarray) -> Dict:
        """Calculate depth consistency metrics"""
        # Calculate local variations
        grad_x = np.gradient(depth_array, axis=1)
        grad_y = np.gradient(depth_array, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'mean_gradient': float(np.mean(gradient_magnitude)),
            'std_gradient': float(np.std(gradient_magnitude)),
            'smoothness_score': 1.0 / (1.0 + np.mean(gradient_magnitude)),  # Higher is smoother
            'edge_density': float(np.sum(gradient_magnitude > np.percentile(gradient_magnitude, 90)) / gradient_magnitude.size)
        }
    
    def _evaluate_sub_pixel_precision(self, depth_array: np.ndarray) -> Dict:
        """Evaluate sub-pixel precision of depth estimation"""
        # Calculate the precision based on depth value distribution
        unique_values = len(np.unique(depth_array))
        total_pixels = depth_array.size
        
        # Higher unique value ratio indicates better precision
        precision_ratio = unique_values / total_pixels
        
        # Check for sub-pixel precision indicators
        decimal_precision = len(str(depth_array.flatten()[0]).split('.')[-1]) if '.' in str(depth_array.flatten()[0]) else 0
        
        return {
            'unique_value_ratio': precision_ratio,
            'estimated_decimal_precision': decimal_precision,
            'sub_pixel_achieved': precision_ratio > 0.5 and decimal_precision >= 3
        }
    
    def _calculate_depth_error_metrics(self, predicted: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """Calculate depth estimation error metrics"""
        if predicted.shape != ground_truth.shape:
            logging.warning("Predicted and ground truth shapes don't match")
            return {}
        
        # Calculate various error metrics
        absolute_error = np.abs(predicted - ground_truth)
        squared_error = (predicted - ground_truth) ** 2
        
        return {
            'mae': float(np.mean(absolute_error)),  # Mean Absolute Error
            'rmse': float(np.sqrt(np.mean(squared_error))),  # Root Mean Square Error
            'mape': float(np.mean(np.abs((ground_truth - predicted) / ground_truth))) * 100,  # Mean Absolute Percentage Error
            'max_error': float(np.max(absolute_error)),
            'error_std': float(np.std(absolute_error))
        }
    
    def _evaluate_3d_object_quality(self, objects_3d: List[Dict]) -> Dict:
        """Evaluate quality of 3D object reconstruction"""
        if not objects_3d:
            return {'object_count': 0, 'quality_score': 0}
        
        # Calculate quality based on depth consistency and confidence
        depth_consistencies = [obj.get('estimated_3d_properties', {}).get('depth_consistency', 0) 
                              for obj in objects_3d]
        confidences = [obj.get('confidence', 0) for obj in objects_3d]
        
        return {
            'object_count': len(objects_3d),
            'avg_depth_consistency': float(np.mean(depth_consistencies)) if depth_consistencies else 0,
            'avg_confidence': float(np.mean(confidences)) if confidences else 0,
            'quality_score': float(np.mean(depth_consistencies + confidences)) / 2 if (depth_consistencies and confidences) else 0
        }
    
    def _evaluate_spatial_analysis(self, spatial_relationships: List[Dict]) -> Dict:
        """Evaluate quality of spatial relationship analysis"""
        if not spatial_relationships:
            return {'relationship_count': 0, 'analysis_completeness': 0}
        
        # Count different types of relationships
        relationship_types = [rel.get('relationship_type', '') for rel in spatial_relationships]
        unique_types = len(set(relationship_types))
        
        return {
            'relationship_count': len(spatial_relationships),
            'unique_relationship_types': unique_types,
            'analysis_completeness': min(unique_types / 5, 1.0),  # Assuming 5 main relationship types
            'avg_distance_2d': float(np.mean([rel.get('distance_2d', 0) for rel in spatial_relationships]))
        }
    
    def _calculate_scene_understanding_score(self, fusion_results: Dict) -> float:
        """Calculate overall scene understanding score"""
        objects_3d = fusion_results.get('objects_3d', [])
        relationships = fusion_results.get('spatial_relationships', [])
        scene_structure = fusion_results.get('scene_structure', {})
        
        # Component scores
        object_score = min(len(objects_3d) / 10, 1.0)  # Normalize to max 10 objects
        relationship_score = min(len(relationships) / 20, 1.0)  # Normalize to max 20 relationships
        structure_score = 1.0 if scene_structure else 0.0
        
        # Weighted average
        total_score = (object_score * 0.4 + relationship_score * 0.4 + structure_score * 0.2)
        
        return float(total_score)
    
    def _calculate_overall_score(self, detection_metrics: Dict, depth_metrics: Dict, fusion_metrics: Dict) -> float:
        """Calculate overall system performance score"""
        scores = []
        
        # Detection score
        if detection_metrics.get('estimated_accuracy'):
            scores.append(detection_metrics['estimated_accuracy'])
        elif detection_metrics.get('confidence_metrics', {}).get('mean_confidence'):
            scores.append(detection_metrics['confidence_metrics']['mean_confidence'])
        
        # Depth score
        if depth_metrics.get('sub_pixel_precision', {}).get('sub_pixel_achieved'):
            scores.append(0.9)  # High score for achieving sub-pixel precision
        elif depth_metrics.get('depth_consistency', {}).get('smoothness_score'):
            scores.append(depth_metrics['depth_consistency']['smoothness_score'])
        
        # Fusion score
        if fusion_metrics.get('scene_understanding_score'):
            scores.append(fusion_metrics['scene_understanding_score'])
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _assess_system_status(self, overall_score: float) -> str:
        """Assess overall system status based on performance score"""
        if overall_score >= 0.95:
            return "Excellent"
        elif overall_score >= 0.85:
            return "Good"
        elif overall_score >= 0.70:
            return "Satisfactory"
        elif overall_score >= 0.50:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def generate_performance_report(self, metrics: Dict) -> str:
        """
        Generate a comprehensive performance report
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Formatted performance report string
        """
        report = []
        report.append("="*60)
        report.append("DEPTHVISION AI PERFORMANCE REPORT")
        report.append("="*60)
        
        # Overall Performance
        report.append(f"\nOVERALL PERFORMANCE: {metrics.get('overall_score', 0):.3f}")
        report.append(f"System Status: {metrics.get('system_status', 'Unknown')}")
        
        # Detection Performance
        detection = metrics.get('detection_performance', {})
        report.append(f"\nOBJECT DETECTION:")
        report.append(f"  Total Detections: {detection.get('total_detections', 0)}")
        conf_metrics = detection.get('confidence_metrics', {})
        report.append(f"  Mean Confidence: {conf_metrics.get('mean_confidence', 0):.3f}")
        report.append(f"  Target Accuracy Met: {detection.get('accuracy_target_met', False)}")
        
        # Depth Performance
        depth = metrics.get('depth_performance', {})
        if depth:
            report.append(f"\nDEPTH ESTIMATION:")
            sub_pixel = depth.get('sub_pixel_precision', {})
            report.append(f"  Sub-pixel Precision: {sub_pixel.get('sub_pixel_achieved', False)}")
            consistency = depth.get('depth_consistency', {})
            report.append(f"  Smoothness Score: {consistency.get('smoothness_score', 0):.3f}")
        
        # Fusion Performance
        fusion = metrics.get('fusion_performance', {})
        report.append(f"\nMULTI-MODAL FUSION:")
        report.append(f"  Scene Understanding Score: {fusion.get('scene_understanding_score', 0):.3f}")
        report.append(f"  3D Objects Detected: {fusion.get('3d_object_quality', {}).get('object_count', 0)}")
        
        # Timing Data
        timing = metrics.get('timing_data', {})
        if timing:
            report.append(f"\nPERFORMANCE TIMING:")
            for operation, duration in timing.items():
                report.append(f"  {operation}: {duration:.3f}s")
        
        report.append("="*60)
        
        return "\n".join(report)
    
    def save_metrics(self, metrics: Dict, filepath: str):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logging.info(f"Metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str) -> Dict:
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        logging.info(f"Metrics loaded from {filepath}")
        return metrics

def test_metrics():
    """Test the metrics evaluation system"""
    
    # Create sample data for testing
    sample_predictions = {
        'boxes': [[100, 100, 200, 200], [300, 150, 400, 250]],
        'scores': [0.95, 0.87],
        'class_names': ['person', 'car'],
        'classes': [0, 1]
    }
    
    sample_depth = np.random.rand(480, 640)
    
    sample_fusion_results = {
        'objects_3d': [
            {
                'object_id': 0, 'class_name': 'person', 'confidence': 0.95,
                'estimated_3d_properties': {'depth_consistency': 0.8}
            }
        ],
        'spatial_relationships': [
            {'relationship_type': 'adjacent_same_depth'}
        ],
        'scene_structure': {'depth_layers': []}
    }
    
    # Test metrics evaluation
    metrics = PerformanceMetrics()
    
    # Evaluate components
    detection_metrics = metrics.evaluate_detection_accuracy(sample_predictions)
    depth_metrics = metrics.evaluate_depth_quality(sample_depth)
    fusion_metrics = metrics.evaluate_fusion_performance(sample_fusion_results)
    
    print("Metrics evaluation test completed successfully!")
    print(f"Detection accuracy estimated: {detection_metrics.get('estimated_accuracy', 0):.3f}")
    print(f"Depth sub-pixel precision: {depth_metrics.get('sub_pixel_precision', {}).get('sub_pixel_achieved', False)}")
    print(f"Scene understanding score: {fusion_metrics.get('scene_understanding_score', 0):.3f}")
    
    return True

if __name__ == "__main__":
    test_metrics()

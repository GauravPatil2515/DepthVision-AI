"""
DepthVision AI - Main Application
Complete pipeline integrating object detection, depth estimation, and multi-modal fusion
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Union
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.depth_estimation import DepthEstimator
from src.models.object_detection import ObjectDetector
from src.models.fusion_model import FusionModel
from src.utils.image_processing import ImageProcessor
from src.utils.visualization import VisualizationTools
from src.utils.metrics import PerformanceMetrics
from src.config.config import config, data_paths

class DepthVisionAI:
    """
    Main DepthVision AI application class
    Orchestrates the complete 3D scene understanding pipeline
    """
    
    def __init__(self, depth_model: str = "Intel/dpt-large", 
                 detection_confidence: float = 0.5,
                 enable_visualization: bool = True,
                 enable_metrics: bool = True):
        """
        Initialize the DepthVision AI system
        
        Args:
            depth_model: Depth estimation model name
            detection_confidence: Object detection confidence threshold
            enable_visualization: Whether to generate visualizations
            enable_metrics: Whether to calculate performance metrics
        """
        # Setup logging
        self._setup_logging()
        
        logging.info("Initializing DepthVision AI System...")
        
        # Initialize core components
        self.fusion_model = FusionModel(
            depth_model_name=depth_model,
            detection_confidence=detection_confidence
        )
        
        # Initialize utilities
        self.image_processor = ImageProcessor()
        
        if enable_visualization:
            self.visualizer = VisualizationTools()
        else:
            self.visualizer = None
            
        if enable_metrics:
            self.metrics_evaluator = PerformanceMetrics()
        else:
            self.metrics_evaluator = None
        
        # System configuration
        self.enable_visualization = enable_visualization
        self.enable_metrics = enable_metrics
        
        logging.info("DepthVision AI System initialized successfully!")
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure logging to both file and console
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(config.LOGGING['log_file']),
                logging.StreamHandler()
            ]
        )
    
    def process_image(self, image_path: Union[str, Path], 
                     output_dir: Optional[Union[str, Path]] = None,
                     save_results: bool = True) -> dict:
        """
        Process a single image through the complete DepthVision AI pipeline
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results (optional)
            save_results: Whether to save results to disk
            
        Returns:
            Complete analysis results dictionary
        """
        logging.info(f"Processing image: {image_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = data_paths.PROCESSED_IMAGES
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate input image
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Validate image quality
        image = self.image_processor.load_image(image_path)
        quality_metrics = self.image_processor.validate_image_quality(image)
        
        logging.info(f"Image quality validation: {quality_metrics}")
        
        # Start comprehensive scene analysis
        start_time = time.time()
        
        if self.metrics_evaluator:
            with self.metrics_evaluator.measure_processing_time("complete_analysis"):
                scene_analysis = self.fusion_model.analyze_scene(image_path)
        else:
            scene_analysis = self.fusion_model.analyze_scene(image_path)
        
        # Calculate performance metrics
        if self.metrics_evaluator:
            performance_metrics = self.metrics_evaluator.evaluate_overall_performance(scene_analysis)
            scene_analysis['performance_metrics'] = performance_metrics
        
        # Generate visualizations
        if self.enable_visualization and self.visualizer:
            self._generate_visualizations(scene_analysis, output_dir, Path(image_path).stem)
        
        # Save results
        if save_results:
            self._save_results(scene_analysis, output_dir, Path(image_path).stem)
        
        processing_time = time.time() - start_time
        logging.info(f"Image processing completed in {processing_time:.3f} seconds")
        
        return scene_analysis
    
    def process_batch(self, input_dir: Union[str, Path], 
                     output_dir: Optional[Union[str, Path]] = None,
                     image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> dict:
        """
        Process a batch of images
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            image_extensions: Supported image file extensions
            
        Returns:
            Batch processing results
        """
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = data_paths.PROCESSED_IMAGES / "batch_results"
        output_dir = Path(output_dir)
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {input_dir}")
        
        logging.info(f"Processing batch of {len(image_files)} images")
        
        batch_results = {
            'total_images': len(image_files),
            'processed_images': [],
            'failed_images': [],
            'batch_statistics': {},
            'processing_times': []
        }
        
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            try:
                logging.info(f"Processing image {i}/{len(image_files)}: {image_file.name}")
                
                start_time = time.time()
                result = self.process_image(
                    image_file, 
                    output_dir / image_file.stem,
                    save_results=True
                )
                processing_time = time.time() - start_time
                
                batch_results['processed_images'].append({
                    'filename': image_file.name,
                    'processing_time': processing_time,
                    'objects_detected': len(result['fusion_analysis']['objects_3d']),
                    'scene_complexity': result['fusion_analysis']['scene_structure']['scene_complexity']
                })
                batch_results['processing_times'].append(processing_time)
                
            except Exception as e:
                logging.error(f"Failed to process {image_file.name}: {e}")
                batch_results['failed_images'].append({
                    'filename': image_file.name,
                    'error': str(e)
                })
        
        # Calculate batch statistics
        if batch_results['processing_times']:
            batch_results['batch_statistics'] = {
                'total_processing_time': sum(batch_results['processing_times']),
                'average_processing_time': sum(batch_results['processing_times']) / len(batch_results['processing_times']),
                'success_rate': len(batch_results['processed_images']) / len(image_files),
                'total_objects_detected': sum(img['objects_detected'] for img in batch_results['processed_images'])
            }
        
        # Save batch results
        self._save_batch_results(batch_results, output_dir)
        
        logging.info(f"Batch processing completed. Success rate: {batch_results['batch_statistics']['success_rate']:.2%}")
        
        return batch_results
    
    def _generate_visualizations(self, scene_analysis: dict, output_dir: Path, image_name: str):
        """Generate and save visualizations"""
        if not self.visualizer:
            return
        
        logging.info("Generating visualizations...")
        
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Generate comprehensive visualization
            main_viz = self.visualizer.visualize_scene_analysis(scene_analysis)
            main_viz.savefig(viz_dir / f"{image_name}_comprehensive.png", 
                           dpi=300, bbox_inches='tight')
            
            # Generate interactive 3D visualization
            interactive_viz = self.visualizer.create_interactive_3d_visualization(scene_analysis)
            interactive_viz.write_html(viz_dir / f"{image_name}_interactive_3d.html")
            
            logging.info(f"Visualizations saved to {viz_dir}")
            
        except Exception as e:
            logging.error(f"Error generating visualizations: {e}")
    
    def _save_results(self, scene_analysis: dict, output_dir: Path, image_name: str):
        """Save analysis results to files"""
        logging.info("Saving analysis results...")
        
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save complete analysis as JSON
            import json
            with open(results_dir / f"{image_name}_analysis.json", 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_safe_analysis = self._make_json_serializable(scene_analysis)
                json.dump(json_safe_analysis, f, indent=2, default=str)
            
            # Save depth map as image
            depth_viz = scene_analysis['depth_estimation']['depth_visualization']
            depth_viz.save(results_dir / f"{image_name}_depth_map.png")
            
            # Save performance report
            if 'performance_metrics' in scene_analysis and self.metrics_evaluator:
                report = self.metrics_evaluator.generate_performance_report(
                    scene_analysis['performance_metrics']
                )
                with open(results_dir / f"{image_name}_performance_report.txt", 'w') as f:
                    f.write(report)
            
            logging.info(f"Results saved to {results_dir}")
            
        except Exception as e:
            logging.error(f"Error saving results: {e}")
    
    def _save_batch_results(self, batch_results: dict, output_dir: Path):
        """Save batch processing results"""
        import json
        
        with open(output_dir / "batch_summary.json", 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        logging.info(f"Batch results saved to {output_dir / 'batch_summary.json'}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-safe formats"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def get_system_info(self) -> dict:
        """Get system information and status"""
        import torch
        
        return {
            'system_version': '1.0.0',
            'pytorch_version': torch.__version__,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'cuda_available': torch.cuda.is_available(),
            'model_configs': {
                'depth_model': config.DEPTH_MODEL,
                'detection_model': config.DETECTION_MODEL
            },
            'performance_targets': {
                'detection_accuracy': '99.94%',
                'depth_precision': 'sub-pixel',
                'max_resolution': config.PROCESSING['max_image_size']
            }
        }

def main():
    """Main entry point for command-line interface"""
    parser = argparse.ArgumentParser(description='DepthVision AI - Object Detection & Depth Estimation')
    
    parser.add_argument('--input', '-i', required=True, 
                       help='Input image file or directory')
    parser.add_argument('--output', '-o', 
                       help='Output directory (default: data/output/processed)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process directory as batch')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization generation')
    parser.add_argument('--no-metrics', action='store_true',
                       help='Disable metrics calculation')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--depth-model', default='Intel/dpt-large',
                       help='Depth estimation model (default: Intel/dpt-large)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize DepthVision AI system
        system = DepthVisionAI(
            depth_model=args.depth_model,
            detection_confidence=args.confidence,
            enable_visualization=not args.no_viz,
            enable_metrics=not args.no_metrics
        )
        
        # Print system information
        print("DepthVision AI - Object Detection & Depth Estimation 2025")
        print("="*60)
        system_info = system.get_system_info()
        for key, value in system_info.items():
            print(f"{key}: {value}")
        print("="*60)
        
        # Process input
        if args.batch:
            results = system.process_batch(args.input, args.output)
            print(f"\nBatch processing completed!")
            print(f"Images processed: {results['batch_statistics']['success_rate']:.1%}")
            print(f"Total objects detected: {results['batch_statistics']['total_objects_detected']}")
        else:
            results = system.process_image(args.input, args.output)
            print(f"\nImage processing completed!")
            print(f"Objects detected: {len(results['fusion_analysis']['objects_3d'])}")
            print(f"Scene complexity: {results['fusion_analysis']['scene_structure']['scene_complexity']}")
            
            if 'performance_metrics' in results:
                print(f"Overall performance score: {results['performance_metrics']['overall_score']:.3f}")
        
        print(f"Results saved to: {args.output or data_paths.PROCESSED_IMAGES}")
        
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

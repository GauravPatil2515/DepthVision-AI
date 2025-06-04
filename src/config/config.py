"""
Configuration settings for DepthVision AI project
Centralized configuration management for models, paths, and parameters
"""

import os
from pathlib import Path

class Config:
    """Main configuration class for the DepthVision AI project"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    INPUT_DIR = DATA_DIR / "input"
    OUTPUT_DIR = DATA_DIR / "output"
    MODELS_DIR = DATA_DIR / "models"
    
    # Ensure directories exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Model configurations
    DEPTH_MODEL = {
        'name': 'Intel/dpt-large',
        'input_size': (384, 384),
        'output_channels': 1
    }
    
    DETECTION_MODEL = {
        'config_file': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        'weights': 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl',
        'confidence_threshold': 0.5,
        'nms_threshold': 0.5
    }
    
    # Processing parameters
    PROCESSING = {
        'max_image_size': (6167, 4632),  # Ultra-high resolution support
        'batch_size': 1,  # For ultra-high res images
        'target_accuracy': 0.9994,  # 99.94% detection accuracy
        'depth_precision': 'sub-pixel'
    }
    
    # Visualization settings
    VISUALIZATION = {
        'depth_colormap': 'plasma',
        'detection_colors': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
        'font_size': 12,
        'line_thickness': 2
    }
    
    # Logging configuration
    LOGGING = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_file': OUTPUT_DIR / 'depthvision.log'
    }
    
    # Device configuration
    DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'auto'
    
    # Performance settings
    PERFORMANCE = {
        'use_mixed_precision': True,
        'optimize_memory': True,
        'parallel_processing': True
    }

class ModelPaths:
    """Paths for storing and loading trained models"""
    
    DEPTH_MODEL_PATH = Config.MODELS_DIR / "depth_estimation_model.pth"
    DETECTION_MODEL_PATH = Config.MODELS_DIR / "detection_model.pth"
    FUSION_MODEL_PATH = Config.MODELS_DIR / "fusion_model.pth"

class DataPaths:
    """Paths for input and output data"""
    
    SAMPLE_IMAGES = Config.INPUT_DIR / "samples"
    PROCESSED_IMAGES = Config.OUTPUT_DIR / "processed"
    DEPTH_MAPS = Config.OUTPUT_DIR / "depth_maps"
    DETECTION_RESULTS = Config.OUTPUT_DIR / "detections"
    FUSION_RESULTS = Config.OUTPUT_DIR / "fusion"
    
    # Create subdirectories
    for path in [SAMPLE_IMAGES, PROCESSED_IMAGES, DEPTH_MAPS, DETECTION_RESULTS, FUSION_RESULTS]:
        path.mkdir(parents=True, exist_ok=True)

# Export configuration instances
config = Config()
model_paths = ModelPaths()
data_paths = DataPaths()

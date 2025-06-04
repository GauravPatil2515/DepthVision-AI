"""
Image Processing Utilities for DepthVision AI
Handles image preprocessing, postprocessing, and format conversions
"""

import cv2
import numpy as np
from PIL import Image
import torch
from typing import Union, Tuple, List
import logging

class ImageProcessor:
    """
    Comprehensive image processing utilities for the DepthVision AI pipeline
    """
    
    @staticmethod
    def load_image(image_path: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """
        Load image from various input formats
        
        Args:
            image_path: Path to image, PIL Image, or numpy array
            
        Returns:
            PIL Image object
        """
        if isinstance(image_path, str):
            return Image.open(image_path).convert('RGB')
        elif isinstance(image_path, np.ndarray):
            return Image.fromarray(image_path)
        elif isinstance(image_path, Image.Image):
            return image_path.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image_path)}")
    
    @staticmethod
    def resize_image(image: Image.Image, target_size: Tuple[int, int], 
                     maintain_aspect_ratio: bool = True) -> Image.Image:
        """
        Resize image while optionally maintaining aspect ratio
        
        Args:
            image: Input PIL Image
            target_size: Target (width, height)
            maintain_aspect_ratio: Whether to maintain original aspect ratio
            
        Returns:
            Resized PIL Image
        """
        if not maintain_aspect_ratio:
            return image.resize(target_size, Image.LANCZOS)
        
        # Calculate new size maintaining aspect ratio
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        
        # Use the smaller ratio to ensure image fits within target size
        ratio = min(width_ratio, height_ratio)
        
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    @staticmethod
    def normalize_image(image: np.ndarray, mean: List[float] = None, 
                       std: List[float] = None) -> np.ndarray:
        """
        Normalize image using mean and standard deviation
        
        Args:
            image: Input image array
            mean: Mean values for normalization (default: ImageNet mean)
            std: Standard deviation values (default: ImageNet std)
            
        Returns:
            Normalized image array
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]  # ImageNet mean
        if std is None:
            std = [0.229, 0.224, 0.225]   # ImageNet std
            
        image = image.astype(np.float32) / 255.0
        
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
            
        return image
    
    @staticmethod
    def denormalize_image(image: np.ndarray, mean: List[float] = None, 
                         std: List[float] = None) -> np.ndarray:
        """
        Denormalize image back to original scale
        
        Args:
            image: Normalized image array
            mean: Mean values used for normalization
            std: Standard deviation values used for normalization
            
        Returns:
            Denormalized image array
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
            
        for i in range(3):
            image[:, :, i] = image[:, :, i] * std[i] + mean[i]
            
        image = (image * 255.0).clip(0, 255).astype(np.uint8)
        return image
    
    @staticmethod
    def enhance_image_quality(image: Image.Image, enhance_contrast: bool = True,
                             enhance_sharpness: bool = True) -> Image.Image:
        """
        Enhance image quality for better processing results
        
        Args:
            image: Input PIL Image
            enhance_contrast: Whether to enhance contrast
            enhance_sharpness: Whether to enhance sharpness
            
        Returns:
            Enhanced PIL Image
        """
        try:
            from PIL import ImageEnhance
            
            if enhance_contrast:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)  # Increase contrast by 20%
            
            if enhance_sharpness:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)  # Increase sharpness by 10%
                
        except ImportError:
            logging.warning("PIL ImageEnhance not available, skipping enhancement")
            
        return image
    
    @staticmethod
    def create_image_grid(images: List[Image.Image], grid_size: Tuple[int, int] = None,
                         image_size: Tuple[int, int] = (256, 256)) -> Image.Image:
        """
        Create a grid of images for visualization
        
        Args:
            images: List of PIL Images
            grid_size: Grid dimensions (rows, cols). Auto-calculated if None
            image_size: Size to resize each image to
            
        Returns:
            PIL Image containing the grid
        """
        if not images:
            raise ValueError("No images provided")
        
        # Auto-calculate grid size if not provided
        if grid_size is None:
            n_images = len(images)
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
            grid_size = (rows, cols)
        
        rows, cols = grid_size
        
        # Resize all images
        resized_images = [img.resize(image_size, Image.LANCZOS) for img in images]
        
        # Create grid
        grid_width = cols * image_size[0]
        grid_height = rows * image_size[1]
        grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        
        for i, img in enumerate(resized_images):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            x = col * image_size[0]
            y = row * image_size[1]
            
            grid_image.paste(img, (x, y))
        
        return grid_image
    
    @staticmethod
    def convert_depth_to_colormap(depth_array: np.ndarray, 
                                 colormap: str = 'plasma') -> np.ndarray:
        """
        Convert depth array to colormap for visualization
        
        Args:
            depth_array: Input depth array
            colormap: OpenCV colormap name
            
        Returns:
            Colored depth image as numpy array
        """
        # Normalize depth values to 0-255
        normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        
        # Apply colormap
        colormap_dict = {
            'plasma': cv2.COLORMAP_PLASMA,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'cool': cv2.COLORMAP_COOL
        }
        
        cm = colormap_dict.get(colormap, cv2.COLORMAP_PLASMA)
        colored = cv2.applyColorMap(normalized, cm)
        
        # Convert BGR to RGB
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def validate_image_quality(image: Image.Image) -> dict:
        """
        Validate image quality and provide metrics
        
        Args:
            image: Input PIL Image
            
        Returns:
            Dictionary with quality metrics
        """
        img_array = np.array(image)
        
        # Calculate basic metrics
        metrics = {
            'width': image.width,
            'height': image.height,
            'channels': len(img_array.shape),
            'mean_brightness': np.mean(img_array),
            'std_brightness': np.std(img_array),
            'min_value': np.min(img_array),
            'max_value': np.max(img_array)
        }
        
        # Check for common issues
        metrics['is_too_dark'] = metrics['mean_brightness'] < 50
        metrics['is_too_bright'] = metrics['mean_brightness'] > 200
        metrics['is_low_contrast'] = metrics['std_brightness'] < 30
        metrics['is_ultra_high_res'] = image.width * image.height > 20_000_000  # 20MP+
        
        return metrics

# Utility function for quick image processing
def process_image_for_model(image_path: str, target_size: Tuple[int, int] = (512, 512)) -> Tuple[Image.Image, np.ndarray]:
    """
    Quick utility to process an image for model input
    
    Args:
        image_path: Path to input image
        target_size: Target size for processing
        
    Returns:
        Tuple of (processed_PIL_image, normalized_array)
    """
    processor = ImageProcessor()
    
    # Load and resize image
    image = processor.load_image(image_path)
    resized_image = processor.resize_image(image, target_size)
    
    # Convert to array and normalize
    img_array = np.array(resized_image)
    normalized_array = processor.normalize_image(img_array)
    
    return resized_image, normalized_array

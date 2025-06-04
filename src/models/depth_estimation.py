"""
Depth Estimation Module using Intel MiDaS DPT-Large
Processes images to generate depth maps with sub-pixel precision
"""

from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import torch
import numpy as np
from typing import Union, Tuple
import logging

class DepthEstimator:
    """
    Intel MiDaS DPT-Large based depth estimation model
    Supports ultra-high-resolution images with sub-pixel depth precision
    """
    
    def __init__(self, model_name: str = "Intel/dpt-large", device: str = None):
        """
        Initialize depth estimation model
        
        Args:
            model_name: Name/path of the model to load
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.logger = logging.getLogger(__name__)
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        try:
            # Load model and processor
            self.processor = DPTImageProcessor.from_pretrained(model_name)
            self.model = DPTForDepthEstimation.from_pretrained(model_name)
            self.model.to(self.device)
            
            self.logger.info(f"Depth estimation model loaded on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load depth model: {e}")
            raise
    
    def estimate_depth(self, image: Union[Image.Image, str, np.ndarray]) -> dict:
        """
        Generate depth map for input image
        
        Args:
            image: Input image (PIL Image, file path, or numpy array)
            
        Returns:
            Dictionary containing:
                - depth_map: Normalized depth values array
                - depth_visualization: Colored depth visualization
                - statistics: Various depth statistics
                - error: Error message if failed, None otherwise
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                input_image = Image.open(image)
            elif isinstance(image, np.ndarray):
                input_image = Image.fromarray(image)
            else:
                input_image = image
                
            # Ensure RGB mode
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            
            # Process image for model
            inputs = self.processor(images=input_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate depth prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                depth_preds = outputs.predicted_depth
            
            # Post-process depth predictions
            depth_array = torch.nn.functional.interpolate(
                depth_preds.unsqueeze(1),
                size=input_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
            
            # Normalize depth values
            depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
            
            # Generate visualization
            depth_image = self._create_depth_visualization(depth_array)
            
            # Calculate statistics
            depth_stats = self.get_depth_statistics(depth_array)
            
            # Free GPU memory
            torch.cuda.empty_cache()  # Free up GPU memory if using CUDA
            
            logging.info("Depth estimation completed successfully")
            result_dict = {
                'depth_map': depth_array,
                'depth_visualization': depth_image,
                'statistics': depth_stats,
                'error': None
            }
            return result_dict
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Depth estimation failed: {error_msg}")
            result_dict = {
                'depth_map': None,
                'depth_visualization': None,
                'statistics': {},
                'error': error_msg
            }
            return result_dict
        finally:
            # Ensure input image is cleaned up
            if 'input_image' in locals():
                del input_image
    
    def _create_depth_visualization(self, depth_array: np.ndarray) -> Image.Image:
        """Convert depth array to color visualization"""
        import matplotlib.pyplot as plt
        
        # Create colormap visualization
        plt.ioff()  # Turn off interactive mode
        fig, ax = plt.subplots()
        img = ax.imshow(depth_array, cmap='plasma')
        plt.colorbar(img)
        
        # Save to bytes buffer
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        
        # Convert to PIL Image
        buf.seek(0)
        depth_viz = Image.open(buf)
        return depth_viz
    
    def get_depth_statistics(self, depth_array: np.ndarray) -> dict:
        """Calculate depth map statistics"""
        return {
            'min_depth': float(depth_array.min()),
            'max_depth': float(depth_array.max()),
            'mean_depth': float(depth_array.mean()),
            'std_depth': float(depth_array.std()),
            'median_depth': float(np.median(depth_array))
        }

# Example usage for testing
def test_depth_estimation():
    """Test function to verify depth estimation functionality"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the estimator
    estimator = DepthEstimator()
    
    # Test with a sample image from COCO dataset
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    try:
        import requests
        from io import BytesIO
        
        # Download and process image
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        
        # Test depth estimation
        results = estimator.estimate_depth(image)
        
        if results['error'] is None:
            print("✅ Depth estimation test passed")
            print(f"Statistics: {results['statistics']}")
        else:
            print(f"❌ Test failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_depth_estimation()

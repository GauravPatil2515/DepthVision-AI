"""
Object Detection Module with multiple backend support
Supports Detectron2 (Mask R-CNN) and YOLO as fallback
"""

import logging
import numpy as np
from PIL import Image
from typing import Union, List, Dict, Tuple
import torch

class ObjectDetector:
    """
    Multi-backend object detection with Detectron2 and YOLO support
    Automatically falls back to available backend
    """
    
    def __init__(self, model_config: str = None, confidence_threshold: float = 0.5):
        """Initialize object detector with specified backend"""
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        try:
            # Try loading Detectron2
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            
            self.logger.info("Using Detectron2 backend")
            
            # Configure Detectron2
            cfg = get_cfg()
            if model_config:
                cfg.merge_from_file(model_config)
            else:
                cfg.merge_from_file(model_zoo.get_config_file(
                    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                ))
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                )
            
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.predictor = DefaultPredictor(cfg)
            self.backend = "detectron2"
            
        except ImportError:
            self.logger.warning("Detectron2 not available, trying YOLO")
            try:
                # Try loading YOLO
                from ultralytics import YOLO
                self.predictor = YOLO('yolov8n.pt')
                self.backend = "yolo"
                self.logger.info("Using YOLO backend")
            except ImportError:
                self.logger.warning("YOLO not available, using dummy detector")
                self.predictor = DummyDetector()
                self.backend = "dummy"
    
    def detect(self, image: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Detect objects in image
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            Dictionary containing:
                - detections: List of detected objects with boxes and scores
                - visualization: Annotated image
                - error: Error message if failed, None otherwise
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                image = Image.open(image)
                
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
                
            # Ensure RGB format
            if len(image_np.shape) == 2:  # Grayscale
                image_np = np.stack([image_np] * 3, axis=-1)
                
            if self.backend == "detectron2":
                return self._detect_detectron2(image_np)
            elif self.backend == "yolo":
                return self._detect_yolo(image_np)
            else:
                return self._detect_dummy(image_np)
                
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Detection failed: {error_msg}")
            return {
                'detections': [],
                'visualization': None,
                'error': error_msg
            }
    
    def _detect_detectron2(self, image_np: np.ndarray) -> Dict:
        """Detect using Detectron2 backend"""
        outputs = self.predictor(image_np)
        
        # Extract predictions
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        # Create detection list
        detections = []
        for box, score, class_id in zip(boxes, scores, classes):
            if score >= self.confidence_threshold:
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': float(score),
                    'class_id': int(class_id),
                    'class_name': self._get_class_name(class_id)
                })
        
        return {
            'detections': detections,
            'visualization': self._create_visualization(image_np, detections),
            'error': None
        }
    
    def _detect_yolo(self, image_np: np.ndarray) -> Dict:
        """Detect using YOLO backend"""
        results = self.predictor(image_np)
        result = results[0]  # Get first result
        
        detections = []
        for box in result.boxes:
            if box.conf >= self.confidence_threshold:
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append({
                    'bbox': xyxy.tolist(),
                    'confidence': float(box.conf),
                    'class_id': int(box.cls),
                    'class_name': self._get_class_name(int(box.cls))
                })
        
        return {
            'detections': detections,
            'visualization': self._create_visualization(image_np, detections),
            'error': None
        }
    
    def _detect_dummy(self, image_np: np.ndarray) -> Dict:
        """Fallback dummy detector"""
        self.logger.warning("Using dummy detector")
        return {
            'detections': [],
            'visualization': None,
            'error': "No detection backend available"
        }
    
    def _get_class_name(self, class_id: int) -> str:
        """Convert class ID to name using COCO classes"""
        # Simplified COCO class list
        coco_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light"]
        if 0 <= class_id < len(coco_classes):
            return coco_classes[class_id]
        return f"class_{class_id}"
    
    def _create_visualization(self, image: np.ndarray, detections: List[Dict]) -> Image.Image:
        """Create visualization of detections"""
        import cv2
        
        # Convert to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw detections
        for det in detections:
            box = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # Convert to integers
            x1, y1, x2, y2 = map(int, box)
            
            # Draw box
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(image_bgr, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert back to RGB and PIL Image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)

class DummyDetector:
    """Dummy detector for testing"""
    def __call__(self, image):
        return {"instances": []}

def test_object_detection():
    """Test the object detector"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize detector
    detector = ObjectDetector()
    
    try:
        # Test with sample image
        import requests
        from io import BytesIO
        
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        
        # Run detection
        results = detector.detect(image)
        
        if results['error'] is None:
            print("✅ Detection test passed")
            print(f"Found {len(results['detections'])} objects")
            for det in results['detections']:
                print(f"- {det['class_name']}: {det['confidence']:.2f}")
        else:
            print(f"❌ Test failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_object_detection()

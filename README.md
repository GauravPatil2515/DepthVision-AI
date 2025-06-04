# DepthVision AI

A sophisticated computer vision project that combines object detection, depth estimation, and 3D scene analysis capabilities. This project provides a complete pipeline for analyzing images with multiple AI backends and a user-friendly Streamlit dashboard.

## 🌟 Features

- **Multi-Backend Object Detection**
  - Primary: Detectron2 (Mask R-CNN)
  - Fallback: YOLOv8
  - Automatic backend selection based on availability
  - Support for COCO classes

- **Depth Estimation**
  - Intel DPT (Dense Prediction Transformer)
  - High-quality monocular depth estimation
  - GPU acceleration support

- **3D Scene Analysis**
  - Fusion of object detection and depth data
  - 3D scene reconstruction
  - Interactive visualizations

- **User Interface**
  - Streamlit dashboard for easy interaction
  - Real-time visualization
  - Batch processing capabilities

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- PyTorch
- OpenCV

### Installation

1. Clone the repository:
```bash
git clone https://github.com/GauravPatil2515/DepthVision-AI.git
cd DepthVision-AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Detectron2 (optional but recommended):
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Running the Application

1. Start the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

2. For command-line usage:
```bash
python src/main.py --input path/to/image.jpg
```

## 📁 Project Structure

```
DepthVision AI/
├── src/
│   ├── models/          # Core AI models
│   ├── utils/           # Helper functions
│   └── config/          # Configuration files
├── data/
│   ├── input/           # Input images
│   ├── output/          # Generated results
│   └── models/          # Pre-trained models
├── tests/               # Unit tests
└── streamlit_app.py     # Web interface
```

## 🔧 Configuration

The project supports various configuration options:

- Object Detection:
  - Model selection (Detectron2/YOLO)
  - Confidence threshold
  - Custom model configurations

- Depth Estimation:
  - Model selection
  - Output resolution
  - Processing quality

All configurations can be modified in `src/config/config.py`.

## 📊 Sample Results

The project generates:
- Object detection boxes with labels and confidence scores
- Depth maps
- 3D scene visualizations
- Performance metrics

Results are saved in the `data/output/` directory.

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Individual component tests:
```bash
python test_object_detection.py
python test_depth_estimation.py
python test_system.py
```

## 📚 API Reference

### Object Detection

```python
from src.models.object_detection import ObjectDetector

detector = ObjectDetector(confidence_threshold=0.5)
results = detector.detect(image_path)
```

### Depth Estimation

```python
from src.models.depth_estimation import DepthEstimator

estimator = DepthEstimator()
depth_map = estimator.estimate(image)
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

- **Gaurav Patil**
  - GitHub: [@GauravPatil2515](https://github.com/GauravPatil2515)

## 🙏 Acknowledgments

- Facebook AI Research for Detectron2
- Ultralytics for YOLOv8
- Intel for DPT
- Streamlit team for the dashboard framework

---

For more detailed information, check out the following guides:
- [Dashboard Guide](DASHBOARD_GUIDE.md)
- [Setup Guide](SETUP_COMPLETE.md)
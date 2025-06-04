# 🔍 DepthVision AI - Interactive 3D Scene Analysis

A comprehensive AI system for analyzing 3D environments using multiple AI models. This system combines object detection, depth estimation, and scene understanding to create rich 3D scene representations with an interactive web dashboard.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-green.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Features

### 🎯 Multi-Backend Object Detection
- **Primary**: Detectron2 (Facebook AI Research) - State-of-the-art detection
- **Fallback**: YOLOv8 (Ultralytics) - Fast and efficient  
- **Backup**: Dummy detector for testing environments
- Support for 80+ COCO object classes
- Real-time confidence scoring and bounding box regression

### 📏 Advanced Depth Estimation
- **Intel DPT** (Dense Prediction Transformer) - High-quality depth maps
- **MiDaS** integration for robust depth estimation
- Real-time depth map generation and statistical analysis
- Sub-pixel depth precision with normalized depth values

### 🔄 3D Scene Fusion
- Combines 2D object detection with depth information
- Creates accurate 3D object positioning in space
- Advanced scene structure analysis and complexity metrics
- Spatial density calculations and depth layer analysis

### 📊 Interactive Web Dashboard
- **Real-time Processing**: Upload images and get instant AI analysis
- **Multiple Input Methods**: File upload, camera capture, URL, sample images
- **Interactive Visualizations**: Plotly-powered charts and 3D scenes
- **Performance Metrics**: Real-time processing statistics
- **Tabbed Interface**: Organized results display

### 🎨 Comprehensive Visualization
- Object detection overlays with confidence scores
- Interactive depth heatmaps with color mapping
- 3D scene plots with object positioning
- Performance analytics and system monitoring

### ⚙️ Flexible Configuration
- YAML-based configuration system
- Runtime parameter adjustment via web interface
- Multiple output formats and export options
- Configurable confidence thresholds and model backends

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/depthvision-ai.git
cd depthvision-ai

# Install dependencies
pip install -r requirements.txt

# Install Detectron2 (for advanced object detection)
pip install -e ./detectron2 --no-build-isolation
```

### 2. Launch Interactive Dashboard

```bash
# Start the Streamlit dashboard
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### 3. Command Line Usage

```bash
# Process a single image
python src/main.py --input data/input/test_image.jpg --output data/output/

# Process with custom settings
python src/main.py --input path/to/image.jpg --confidence 0.7 --save-results
```

## 🖥️ Dashboard Usage

### Image Input Options
1. **Upload Image**: Drag & drop or browse for JPG/PNG files
2. **Camera Capture**: Take photos directly from your webcam
3. **URL Input**: Process images from web URLs
4. **Sample Images**: Use pre-loaded demonstration images

### Analysis Features
- **Object Detection Tab**: View detected objects with confidence scores
- **Depth Analysis Tab**: Explore depth maps and statistics
- **3D Fusion Tab**: Interactive 3D scene reconstruction
- **Raw Data Tab**: Access detailed JSON results

### Configuration Options
- Adjust detection confidence thresholds
- Enable/disable result saving
- View system information and performance metrics

## 📁 Project Structure

```
DepthVision AI/
├── streamlit_app.py          # Interactive web dashboard
├── src/
│   ├── main.py              # Command-line interface
│   ├── models/
│   │   ├── fusion_model.py  # Main orchestrator class
│   │   ├── object_detection.py  # Multi-backend detector
│   │   └── depth_estimation.py  # Depth estimation engine
│   ├── utils/
│   │   ├── visualization.py # Plotting and visualization
│   │   └── metrics.py       # Performance metrics
│   └── config/
│       └── config.py        # Configuration management
├── data/
│   ├── input/
│   │   ├── samples/         # Sample images for testing
│   │   └── test_image.jpg   # Default test image
│   └── output/              # Processing results
├── detectron2/              # Detectron2 source installation
├── tests/                   # Test suites
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🧪 Testing

```bash
# Run comprehensive system test
python test_system.py

# Test specific components
python -m pytest tests/
```

## 📊 Performance Metrics

The system tracks and displays:
- **Detection Accuracy**: Object detection precision
- **Processing Time**: End-to-end analysis duration
- **Objects Detected**: Count of identified objects
- **Scene Complexity**: Calculated complexity score
- **Memory Usage**: System resource utilization

## 🔧 Configuration

Edit `src/config/config.py` to customize:

```python
# Object Detection Settings
DETECTION_CONFIDENCE_THRESHOLD = 0.5
MAX_DETECTIONS_PER_IMAGE = 100

# Depth Estimation Settings  
DEPTH_MODEL = "intel/dpt-large"
DEPTH_RESOLUTION = (518, 518)

# Output Settings
SAVE_INTERMEDIATE_RESULTS = True
OUTPUT_FORMATS = ["json", "png", "jpg"]
```

## 🎯 Model Backends

### Object Detection
1. **Detectron2** (Primary)
   - Model: Mask R-CNN with ResNet-50 backbone
   - Dataset: COCO 2017
   - Classes: 80 object categories
   - Accuracy: ~40 mAP

2. **YOLOv8** (Fallback)
   - Model: YOLOv8n/s/m/l/x variants
   - Fast inference for real-time applications
   - Configurable model size vs accuracy trade-off

### Depth Estimation
1. **Intel DPT-Large**
   - Transformer-based architecture
   - High-quality depth prediction
   - Robust to various scene types

2. **MiDaS**
   - Fallback option for compatibility
   - Reliable depth estimation
   - Good performance on diverse images

## 🐛 Troubleshooting

### Common Issues

**Detectron2 Installation Fails**
```bash
# Install build tools first
pip install --upgrade pip setuptools wheel
# Then install detectron2
pip install -e ./detectron2 --no-build-isolation
```

**Out of Memory Errors**
- Reduce image resolution in config
- Use smaller model variants (YOLOv8n instead of YOLOv8x)
- Close other applications to free memory

**Slow Processing**
- Enable GPU acceleration if available
- Use smaller input images
- Adjust confidence thresholds to reduce detections

## 📈 Results Examples

### Object Detection
- Detects common objects: person, car, bicycle, dog, cat, etc.
- Confidence scores typically 0.5+ for reliable detections
- Bounding box coordinates in (x1, y1, x2, y2) format

### Depth Analysis
- Normalized depth values (0.0 = closest, 1.0 = farthest)
- Statistical analysis: min, max, mean, standard deviation
- Interactive depth map visualization

### 3D Scene Fusion
- Objects positioned in 3D space using depth information
- Scene complexity scoring based on object count and distribution
- Spatial density analysis for scene understanding

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Detectron2**: Facebook AI Research
- **Intel DPT**: Intel Labs
- **MiDaS**: Intel ISL
- **Streamlit**: Streamlit Inc.
- **YOLOv8**: Ultralytics

## 📞 Support

For questions and support:
- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/depthvision-ai/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/depthvision-ai/wiki)

---

<div align="center">
  <strong>🔍 DepthVision AI - Where Computer Vision Meets 3D Understanding</strong>
</div>

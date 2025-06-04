# 🎯 DepthVision AI Dashboard - Quick Start Guide

## 🚀 Getting Started

### Step 1: Launch the Dashboard
```bash
streamlit run streamlit_app.py
```
Open your browser to: `http://localhost:8501`

### Step 2: Choose Your Input Method

#### 📁 Upload Image
- Click "Choose an image file"
- Select JPG, PNG, or BMP files
- Supports high-resolution images

#### 📷 Camera Capture  
- Click "Take a picture"
- Allow camera permissions
- Capture photos directly from webcam

#### 🌐 URL Input
- Enter image URL in text field
- Supports direct links to images
- Example: `https://example.com/image.jpg`

#### 🖼️ Sample Images
- Pre-loaded demonstration images
- Various scene types for testing
- Downloaded automatically on first run

### Step 3: Configure Settings

#### ⚙️ Sidebar Options
- **Confidence Threshold**: Adjust object detection sensitivity (0.1-1.0)
- **Save Results**: Enable/disable saving processed results to disk
- **System Info**: View current system configuration and model status

### Step 4: Analyze Your Image

1. **Select Image**: Choose using any input method above
2. **Review Image Info**: Check image size, format, and source
3. **Click "🚀 Analyze Image"**: Start the AI processing
4. **Watch Progress**: Real-time progress tracking with status updates

### Step 5: Explore Results

#### 📊 Performance Metrics
- **Detection Accuracy**: How well objects were detected
- **Processing Time**: Total analysis duration  
- **Objects Detected**: Number of objects found
- **Scene Complexity**: Calculated complexity score

#### 🎯 Object Detection Tab
- **Interactive Chart**: Bar chart of detected objects with confidence scores
- **Detection Table**: Detailed list with bounding box coordinates
- **Visual Overlay**: Objects highlighted on original image

#### 📏 Depth Analysis Tab  
- **Interactive Depth Map**: Color-coded depth visualization
- **Depth Statistics**: Min, max, mean, and standard deviation
- **Depth Profile**: Cross-section analysis of scene depth

#### 🔄 3D Fusion Tab
- **3D Scene Plot**: Interactive 3D visualization of detected objects
- **Scene Structure**: Complexity, depth layers, and spatial density metrics
- **Object Positioning**: 3D coordinates for each detected object

#### 📋 Raw Data Tab
- **JSON Results**: Complete analysis results in structured format
- **Export Options**: Download results for further analysis
- **Debugging Info**: Detailed processing information

## 🎨 Visualization Features

### Interactive Controls
- **Zoom**: Mouse wheel or zoom controls
- **Pan**: Click and drag to move around
- **Rotate**: 3D plots support rotation and viewing angles
- **Hover Info**: Detailed information on mouse hover

### Color Schemes
- **Detection**: Confidence-based color coding
- **Depth**: Viridis colormap (dark = close, bright = far)
- **3D Scene**: Object-based color differentiation

## 🔧 Advanced Usage

### Custom Configuration
- Modify `src/config/config.py` for system-wide settings
- Adjust model parameters for specific use cases
- Configure output formats and file locations

### Batch Processing
- Use command-line interface for multiple images
- Set up automated processing pipelines
- Export results in various formats (JSON, PNG, CSV)

### Model Selection
- Primary: Detectron2 (highest accuracy)
- Fallback: YOLOv8 (faster processing)
- Backup: Dummy detector (testing/demo)

## 🐛 Troubleshooting

### Dashboard Won't Start
```bash
# Check Streamlit installation
pip install streamlit

# Verify all dependencies
pip install -r requirements.txt
```

### Slow Processing
- Reduce image size before upload
- Lower confidence threshold
- Close other applications to free memory

### No Objects Detected
- Lower confidence threshold (try 0.3-0.4)
- Check image quality and lighting
- Verify image contains recognizable objects

### Memory Errors
- Use smaller input images
- Restart the dashboard
- Check available system memory

## 📸 Best Practices

### Image Selection
- **Good Lighting**: Well-lit scenes work best
- **Clear Objects**: Avoid heavily occluded objects
- **Reasonable Size**: 800x600 to 1920x1080 optimal
- **Common Objects**: COCO dataset classes work best

### Performance Optimization
- Start with sample images to test setup
- Use appropriate confidence thresholds
- Monitor system resources during processing
- Save results only when needed

## 🎯 Example Workflows

### 1. Room Analysis
- Upload interior scene photo
- Set confidence to 0.5
- Analyze furniture and objects
- Export 3D scene layout

### 2. Outdoor Scene
- Use camera capture for outdoor scene
- Detect vehicles, people, signs
- Analyze depth for navigation
- Review spatial relationships

### 3. Object Inventory
- Upload workspace/storage image
- Lower confidence to catch more objects
- Export detection table
- Use for inventory management

## 📊 Understanding Results

### Detection Confidence
- **0.9-1.0**: Very confident detection
- **0.7-0.9**: Good detection
- **0.5-0.7**: Moderate confidence
- **0.3-0.5**: Lower confidence (may be false positive)

### Depth Values
- **0.0**: Closest objects to camera
- **0.5**: Middle distance
- **1.0**: Farthest objects from camera
- **Relative**: Values are normalized, not absolute distances

### Scene Complexity
- **Low (1-3)**: Simple scenes with few objects
- **Medium (4-6)**: Moderate complexity
- **High (7-10)**: Complex scenes with many objects

---

💡 **Tip**: Start with the sample images to familiarize yourself with the interface before processing your own images!

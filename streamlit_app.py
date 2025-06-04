"""
DepthVision AI - Streamlit Dashboard
Interactive web interface for real-time object detection and depth estimation
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import io
import base64
from pathlib import Path
import time
import logging
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="DepthVision AI Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
try:
    from src.simple_ai import DepthVisionAI
    from src.models.object_detection import ObjectDetector
    from src.models.depth_estimation import DepthEstimator
    from src.utils.visualization import VisualizationTools
    from src.utils.metrics import PerformanceMetrics
    from src.config.config import config
except ImportError as e:
    # Fallback to basic imports
    try:
        from src.simple_ai import DepthVisionAI
        st.warning("⚠️ Using simplified AI system. Some advanced features may be limited.")
    except ImportError as e2:
        st.error(f"Failed to import modules: {e2}")
        st.stop()

# Cache the model loading
@st.cache_resource
def load_models():
    """Load and cache the AI models"""
    try:
        with st.spinner("Loading AI models... This may take a few minutes on first run."):
            system = DepthVisionAI()
            return system
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None

@st.cache_data
def load_sample_images():
    """Load sample images for demonstration"""
    sample_dir = Path("data/input/samples")
    samples = []
    
    if sample_dir.exists():
        for img_path in sample_dir.glob("*.jpg"):
            samples.append(str(img_path))
    
    # Add some default URLs if no local samples
    if not samples:
        samples = [
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "http://images.cocodataset.org/val2017/000000397133.jpg",
            "http://images.cocodataset.org/val2017/000000037777.jpg"
        ]
    
    return samples

def process_image_with_progress(system, image, save_results=False):
    """Process image with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    temp_file = None
    
    try:
        # Step 1: Object Detection
        status_text.text("🔍 Detecting objects...")
        progress_bar.progress(25)
        
        # Step 2: Depth Estimation
        status_text.text("📏 Estimating depth...")
        progress_bar.progress(50)
        
        # Step 3: 3D Fusion
        status_text.text("🔄 Fusing 3D data...")
        progress_bar.progress(75)
        
        # Process the image
        if isinstance(image, str):
            results = system.process_image(image, save_results=save_results)
        else:
            try:
                # Create temporary file with proper cleanup
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    temp_file = tmp.name
                    # Convert to RGB to ensure JPEG compatibility
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(temp_file, 'JPEG', quality=95)
                
                # Process after file is saved and closed
                results = system.process_image(temp_file, save_results=save_results)
            
            finally:
                # Clean up temporary file
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception as e:
                        logging.warning(f"Failed to delete temporary file {temp_file}: {e}")
        
        # Step 4: Complete
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        time.sleep(0.5)  # Show completion briefly
        progress_bar.empty()
        status_text.empty()
        
        return results
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        logging.error(f"Processing failed: {e}")
        raise e
    finally:
        # Ensure temp file cleanup in case of early exit
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logging.warning(f"Failed to delete temporary file in cleanup: {e}")

def create_detection_plot(results):
    """Create interactive detection results plot"""
    if not results['object_detection']['detections']:
        return None
    
    detections = results['object_detection']['detections']
    
    # Create DataFrame for plotting
    detection_data = []
    for det in detections:
        detection_data.append({
            'Class': det['class_name'],
            'Confidence': det['confidence'],
            'BBox': f"{det['bbox'][0]:.0f},{det['bbox'][1]:.0f},{det['bbox'][2]:.0f},{det['bbox'][3]:.0f}"
        })
    
    df = pd.DataFrame(detection_data)
    
    # Create bar chart
    fig = px.bar(
        df, 
        x='Class', 
        y='Confidence',
        title="Object Detection Results",
        color='Confidence',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Object Class",
        yaxis_title="Confidence Score"
    )
    
    return fig

def create_depth_plot(depth_map):
    """Create interactive depth visualization"""
    if depth_map is None:
        return None
    
    try:
        fig = px.imshow(
            depth_map,
            color_continuous_scale='viridis',
            title="Depth Map Visualization",
            labels={'color': 'Depth (relative)'}
        )
        
        fig.update_layout(height=500)
        return fig
    except Exception as e:
        logging.error(f"Failed to create depth plot: {e}")
        return None

def create_3d_scene_plot(results):
    """Create 3D scene visualization"""
    if not results['fusion_analysis']['objects_3d']:
        return None
    
    objects_3d = results['fusion_analysis']['objects_3d']
    
    fig = go.Figure()
    
    for obj in objects_3d:
        # Extract 3D coordinates
        x, y, z = obj['position_3d']
        
        fig.add_trace(go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode='markers+text',
            marker=dict(
                size=10,
                color=obj['depth_value'],
                colorscale='viridis',
                showscale=True
            ),
            text=obj['class_name'],
            textposition="top center",
            name=obj['class_name']
        ))
    
    fig.update_layout(
        title="3D Scene Reconstruction",
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Depth"
        ),
        height=600
    )
    
    return fig

def display_performance_metrics(results):
    """Display performance metrics in columns"""
    if 'performance_metrics' not in results:
        return
    
    metrics = results['performance_metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Detection Accuracy",
            f"{metrics['detection_accuracy']:.1%}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Processing Time",
            f"{metrics['processing_time']:.2f}s",
            delta=None
        )
    
    with col3:
        st.metric(
            "Objects Detected",
            metrics['objects_detected'],
            delta=None
        )
    
    with col4:
        st.metric(
            "Scene Complexity",
            metrics['scene_complexity'],
            delta=None
        )

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🔍 DepthVision AI Dashboard")
    st.markdown("""
    **Real-time Object Detection & Depth Estimation**
    
    Upload an image or select a sample to see AI-powered 3D scene analysis with:
    - 🎯 Object Detection (Detectron2)
    - 📏 Depth Estimation (Intel DPT)
    - 🔄 3D Scene Fusion
    - 📊 Performance Analytics
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model settings
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    save_results = st.sidebar.checkbox("Save Results to Disk", value=False)
    
    # Load models
    system = load_models()
    if system is None:
        st.error("Failed to load AI models. Please check the installation.")
        return
    
    # Update confidence threshold
    system.object_detector.confidence_threshold = confidence_threshold
    
    # Display system info
    with st.sidebar.expander("🖥️ System Information"):
        sys_info = system.get_system_info()
        st.json(sys_info)
    
    # Image input methods
    st.header("📸 Image Input")
    
    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Sample Images", "Camera Capture", "URL"]
    )
    
    image = None
    image_source = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload JPG, PNG, or BMP images"
        )
        
        if uploaded_file is not None:
            import io
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image_source = f"Uploaded: {uploaded_file.name}"
    
    elif input_method == "Sample Images":
        samples = load_sample_images()
        
        if samples:
            selected_sample = st.selectbox(
                "Select a sample image:",
                samples
            )
            
            if selected_sample:
                if selected_sample.startswith('http'):
                    try:
                        import requests
                        response = requests.get(selected_sample)
                        image = Image.open(io.BytesIO(response.content))
                        image_source = f"Sample: {selected_sample}"
                    except Exception as e:
                        st.error(f"Failed to load sample image: {e}")
                else:
                    image = Image.open(selected_sample)
                    image_source = f"Sample: {Path(selected_sample).name}"
    
    elif input_method == "Camera Capture":
        camera_input = st.camera_input("Take a picture")
        if camera_input is not None:
            image = Image.open(camera_input)
            image_source = "Camera capture"
    
    elif input_method == "URL":
        image_url = st.text_input(
            "Enter image URL:",
            placeholder="https://example.com/image.jpg"
        )
        
        if image_url:
            try:
                import requests
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content))
                image_source = f"URL: {image_url}"
            except Exception as e:
                st.error(f"Failed to load image from URL: {e}")
    
    # Process image if available
    if image is not None:        # Display input image
        st.header("🖼️ Input Image")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption=image_source, use_container_width=True)
        
        with col2:
            st.info(f"""
            **Image Info:**
            - Size: {image.size[0]} × {image.size[1]}
            - Mode: {image.mode}
            - Source: {image_source}
            """)
        
        # Process button
        if st.button("🚀 Analyze Image", type="primary"):
            try:
                # Process the image
                results = process_image_with_progress(system, image, save_results)
                
                # Store results in session state
                st.session_state['results'] = results
                st.session_state['processed_image'] = image
                
                st.success("✅ Image analysis completed!")
                
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")
                logging.error(f"Processing error: {e}")    # Display results if available
    if 'results' in st.session_state and 'processed_image' in st.session_state:
        results = st.session_state['results']
        
        # Performance metrics
        st.header("📊 Performance Metrics")
        display_performance_metrics(results)
        
        # Results tabs
        st.header("🔍 Analysis Results")
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Object Detection", 
            "📏 Depth Analysis", 
            "🔄 3D Fusion", 
            "📋 Raw Data"
        ])
        
        # Object Detection Tab
        with tab1:
            st.subheader("Object Detection Results")
            if results.get('object_detection', {}).get('detections'):
                # Detection plot
                detection_plot = create_detection_plot(results)
                if detection_plot:
                    st.plotly_chart(detection_plot, use_container_width=True)
                
                # Detection table
                detection_data = []
                for det in results['object_detection']['detections']:
                    detection_data.append({
                        'Class': det['class_name'],
                        'Confidence': f"{det['confidence']:.3f}",
                        'Bounding Box': f"({det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f})"
                    })
                df = pd.DataFrame(detection_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No objects detected. Try lowering the confidence threshold.")
        
        # Depth Analysis Tab
        with tab2:
            st.subheader("Depth Estimation Results")
            depth_results = results.get('depth_estimation', {})
            depth_map = depth_results.get('depth_map')
            if depth_map is not None and isinstance(depth_map, np.ndarray) and depth_map.size > 0:
                # Depth visualization
                depth_plot = create_depth_plot(depth_map)
                if depth_plot is not None:
                    st.plotly_chart(depth_plot, use_container_width=True)
                
                # Depth statistics
                depth_stats = depth_results.get('statistics', {})
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Min Depth", f"{depth_stats.get('min_depth', 0):.2f}")
                    st.metric("Max Depth", f"{depth_stats.get('max_depth', 0):.2f}")
                with col2:
                    st.metric("Mean Depth", f"{depth_stats.get('mean_depth', 0):.2f}")
                    st.metric("Std Depth", f"{depth_stats.get('std_depth', 0):.2f}")
            else:
                st.warning("No depth data available for visualization.")
        
        # 3D Fusion Tab
        with tab3:
            st.subheader("3D Scene Fusion")
            fusion_data = results.get('fusion_analysis', {})
            if fusion_data.get('objects_3d'):
                # 3D scene plot
                scene_plot = create_3d_scene_plot(results)
                if scene_plot:
                    st.plotly_chart(scene_plot, use_container_width=True)
                
                # Scene structure info
                scene_structure = fusion_data.get('scene_structure', {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Scene Complexity", scene_structure.get('scene_complexity', 0))
                with col2:
                    st.metric("Depth Layers", scene_structure.get('depth_layers', 0))
                with col3:
                    st.metric("Spatial Density", f"{scene_structure.get('spatial_density', 0):.3f}")
            else:
                st.warning("No 3D objects detected for fusion analysis.")        # Raw Data Tab
        with tab4:
            st.subheader("Raw Analysis Data")
            
            # Expandable sections for raw data
            with st.expander("Object Detection Data"):
                st.json(results.get('object_detection', {}))
            
            with st.expander("Depth Estimation Data"):
                # Convert numpy arrays to lists for JSON display
                depth_data = results.get('depth_estimation', {}).copy()
                if 'depth_map' in depth_data:
                    depth_data['depth_map'] = "Array data (not displayed)"
                st.json(depth_data)
            
            with st.expander("Fusion Analysis Data"):
                st.json(results.get('fusion_analysis', {}))
            
            with st.expander("Performance Metrics"):
                if 'performance_metrics' in results:
                    st.json(results['performance_metrics'])

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🔍 <strong>DepthVision AI Dashboard</strong> - Powered by Detectron2 & Intel DPT</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main()

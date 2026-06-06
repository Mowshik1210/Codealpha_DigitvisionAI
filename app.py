import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import plotly.graph_objects as go
from PIL import Image

# ====================================================================
# 1. PLATFORM CONFIGURATION & STYLE DESIGN PIPELINE
# ====================================================================
st.set_page_config(
    page_title="DigitVision AI // Multi-Inference Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background: radial-gradient(circle at 80% 0%, #1a1636 0%, #08090f 50%, #030305 100%) !important;
        color: #F1F5F9 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 16px !important;
    }
     /* Custom glassy panels */
    .saas-container {
        background: rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7);
    }
    
    .sidebar-card {
    background: linear-gradient(
        135deg,
        rgba(56, 189, 248, 0.12),
        rgba(139, 92, 246, 0.12)
    );
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
    }
    
    /* Hyper-gradient typography styles */
    .gradient-title {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 52px;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.15;
        margin-bottom: 8px;
    }
    
    .section-subtitle {
        color: #94A3B8;
        font-size: 18px;
        font-weight: 400;
        margin-bottom: 32px;
        letter-spacing: -0.01em;
    }
    
    /* Interactive status components */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 100px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        background: rgba(56, 189, 248, 0.1);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.2);
        margin-bottom: 20px;
    }
    
    /* Output Analytics Modules */
    .prediction-display-card {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.07) 0%, rgba(129, 140, 248, 0.07) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 0 50px rgba(99, 102, 241, 0.1);
    }
    
    .giant-digit {
        font-size: 120px;
        font-weight: 900;
        background: linear-gradient(180deg, #FFFFFF 30%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin: 12px 0;
        filter: drop-shadow(0 0 30px rgba(255,255,255,0.15));
    }
    
    /* Fine-grain custom tables / rows */
    .dense-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 18px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        transition: background 0.2s ease;
    }
    .dense-row:hover {
        background: rgba(255, 255, 255, 0.04);
    }
    .dense-label {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: #CBD5E1;
    }
    .dense-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: #38BDF8;
    }
    
    /* Feature Grid Matrix */
    .feature-matrix {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px;
        margin-top: 16px;
    }
    .feature-node {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 16px;
    }
    .feature-node strong {
        color: #38BDF8;
        display: block;
        margin-bottom: 4px;
        font-size: 15px;
    }
    .feature-node span {
        color: #94A3B8;
        font-size: 13px;
    }
    
    /* Clean overrides for upload mechanisms */
    [data-testid="stFileUploader"] {
        padding: 12px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 16px;
        border: 1px dashed rgba(255, 255, 255, 0.1);
    }

    .app-title {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 44px;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 2px;
    }
    
    .app-subtitle {
        color: #94A3B8;
        font-size: 15px;
        margin-bottom: 28px;
    }
    

    .status-node {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        background: rgba(56, 189, 248, 0.1);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }

    .sidebar-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 16px;
        margin-top: 12px;
    }

    .segmented-digit-badge {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px;
        margin-right: 8px;
        margin-bottom: 8px;
        min-width: 75px;
        text-align: center;
    }

    .master-string-display {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(129, 140, 248, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        margin-bottom: 16px;
    }

    .master-string-value {
        font-size: 64px;
        font-weight: 900;
        letter-spacing: 4px;
        background: linear-gradient(180deg, #FFFFFF 30%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }

    [data-testid="stFileUploader"] {
        padding: 6px;
        background: rgba(0, 0, 0, 0.15);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# 2. CACHED DEEP LEARNING INFERENCE RUNTIME
# ====================================================================
@st.cache_resource
def load_nn_model():
    return tf.keras.models.load_model("model.h5")

try:
    model = load_nn_model()
except Exception as e:
    st.error("Execution Alert: Could not find trained 'model.h5' parameters. Please compile train_model.py first.")

# ==========================================
# 3. SIDE PANEL SYSTEM BRANDING
# ==========================================
with st.sidebar:
    st.markdown("<div class='status-pill'>● AI Engine Model</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-weight:800; letter-spacing:-0.03em; color:#FFF; margin-bottom:4px;'>DigitVision AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:14px; margin-bottom:28px;'>Handwritten Digit Recognition By MOWSHIK G </p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:14px; margin-bottom:28px;'>CSE(AI & ML) ENGINEER in KPRIET</p>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:12px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.05em;'>Pipeline Configuration</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='sidebar-card'>
        <span style='font-size:12px; color:#64748B; display:block;'>CORE ARCHITECTURE</span>
        <span style='font-weight:600; color:#F1F5F9; font-size:14px;'>Convolutional Neural Net</span>
        <div style='height:12px;'></div>
        <span style='font-size:12px; color:#64748B; display:block;'>DATASET ARTIFACT</span>
        <span style='font-weight:600; color:#F1F5F9; font-size:14px;'>MNIST Handwritten Digits</span>
        <div style='height:12px;'></div>
        <span style='font-size:12px; color:#64748B; display:block;'>BENCHMARK ACCURACY</span>
        <span style='font-weight:700; color:#10B981; font-size:14px;'>~99.00% Validated</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='sidebar-card'>
        <span style='font-size:12px; color:#64748B; display:block; margin-bottom:6px;'>TECHNOLOGY STACK</span>
        <code style='color:#38BDF8; font-size:12px; background:none; padding:0;'>• Python v3.11+</code><br>
        <code style='color:#38BDF8; font-size:12px; background:none; padding:0;'>• TensorFlow API</code><br>
        <code style='color:#38BDF8; font-size:12px; background:none; padding:0;'>• OpenCV / Pillow</code><br>
        <code style='color:#38BDF8; font-size:12px; background:none; padding:0;'>• Streamlit UI Matrix</code>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; color:#334155; font-size:11px; margin-top:40px;'>MOWSHIK G // 2026</p>", unsafe_allow_html=True)
# ====================================================================
# 4. MAIN TELEMETRY DASHBOARD MATRIX
# ====================================================================
st.markdown("<h1 style='font-weight:800; letter-spacing:-0.04em; color:#FFF; margin-bottom:4px;'>DigitVision AI</h1>", unsafe_allow_html=True)
st.markdown("<h1 class='app-title'>Multi-Digit Handwritten Recognition AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>An Intelligent AI system that accurately predicts handwritten digits from uploaded images using advanced convolutional neural networks and deep learning algorithms.</p>", unsafe_allow_html=True)

col_input, col_analytics = st.columns([5, 6], gap="large")

with col_input:
    with st.container(border=True):
        st.markdown("<h3 style='font-size:18px; font-weight:700; margin-top:0;'>📤 Multi-Digit Stream Input</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94A3B8; font-size:13px; margin-bottom:16px;'>Upload an image containing single, double, or multi-digit handwritten sequences.</p>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Ingest Document Stream", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        
        if uploaded_file:
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            pil_image = Image.open(uploaded_file)
            
            # --- COMPUTER VISION ADVANCED MULTI-SEGMENTATION LAYER ---
            # Transform to OpenCV image matrix coordinates
            cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Clear noise and use adaptive inverted thresholds
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Check for background inversion anomalies
            # If the image background is primarily white, thresholding flips appropriately
            if np.mean(thresh) > 127:
                thresh = cv2.bitwise_not(thresh)
                
            # Detect contours (individual character boundary boxes)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            digit_bounding_boxes = []
            for ctr in contours:
                x, y, w, h = cv2.boundingRect(ctr)
                # Ignore small isolated dot artifacts
                if w >= 4 and h >= 12:
                    digit_bounding_boxes.append((x, y, w, h))
            
            # Sort individual digit bounding boxes cleanly from left-to-right
            digit_bounding_boxes = sorted(digit_bounding_boxes, key=lambda b: b[0])
            
            # Draw visual telemetry indicators directly onto bounding canvases
            annotated_img = cv_img.copy()
            processed_digits = []
            
            for index, (x, y, w, h) in enumerate(digit_bounding_boxes):
                # Draw sharp targeting rect squares around inputs
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (56, 189, 248), 2)
                cv2.putText(annotated_img, f"#{index+1}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (129, 140, 248), 1)
                
                # Extract localized character matrix segments
                extracted_roi = thresh[y:y+h, x:x+w]
                
                # Pad out cropped segments uniformly to protect aspect ratios
                pad_dim = max(w, h) + 14
                square_canvas = np.zeros((pad_dim, pad_dim), dtype=np.uint8)
                dx = (pad_dim - w) // 2
                dy = (pad_dim - h) // 2
                square_canvas[dy:dy+h, dx:dx+w] = extracted_roi
                
                # Rescale input cleanly down to canonical MNIST sizes (28x28)
                digit_matrix = cv2.resize(square_canvas, (28, 28), interpolation=cv2.INTER_AREA)
                digit_matrix = digit_matrix / 255.0
                digit_matrix = digit_matrix.reshape(1, 28, 28, 1)
                
                # Evaluate classifications against deep neural nodes
                prediction = model.predict(digit_matrix)
                predicted_lbl = np.argmax(prediction)
                confidence_val = np.max(prediction) * 100
                
                processed_digits.append({
                    "label": predicted_lbl,
                    "confidence": confidence_val,
                    "probabilities": prediction[0] * 100,
                    "crop": digit_matrix.reshape(28, 28)
                })
                
            st.image(annotated_img, caption="Dynamic Computer Vision Segmentation Viewport", use_container_width=True)
            
        else:
            st.markdown("""
            <div style='text-align:center; padding:54px 16px; border:1px dashed rgba(255,255,255,0.08); border-radius:12px; background:rgba(0,0,0,0.1); margin-top:10px;'>
                <p style='font-size:36px; margin-bottom:8px;'>🔮</p>
                <h5 style='color:#E2E8F0; margin-bottom:4px; font-size:15px; font-weight:600;'>Awaiting Character Stream</h5>
                <p style='color:#64748B; font-size:12px; max-width:280px; margin:0 auto;'>Feed multi-digit numerical images to initiate contour sorting structures.</p>
            </div>
            """, unsafe_allow_html=True)

with col_analytics:
    if uploaded_file and len(processed_digits) > 0:
        # Build Master Result String Array
        full_number_string = "".join([str(d["label"]) for d in processed_digits])
        
        st.markdown("<div class='master-string-display'>", unsafe_allow_html=True)
        st.markdown("<span style='color:#38BDF8; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; display:block; margin-bottom:4px;'>Resolved Sequence String</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='master-string-value'>{full_number_string}</div>", unsafe_allow_html=True)
        st.markdown(f"<span style='font-size:13px; color:#10B981;'>Parsed {len(processed_digits)} individual character matrix tracks</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Grid layout for broken down sub-characters
        with st.container(border=True):
            st.markdown("<h4 style='margin-top:0; font-size:15px; font-weight:700;'>🎯 Granular Segment Evaluations</h4>", unsafe_allow_html=True)
            
            # Render visual matrix nodes for extracted bounding targets
            st.markdown("<div style='display: flex; flex-wrap: wrap;'>", unsafe_allow_html=True)
            for i, d in enumerate(processed_digits):
                st.markdown(f"""
                <div class='segmented-digit-badge'>
                    <span style='font-size:11px; color:#64748B; font-weight:700;'>SEGMENT #{i+1}</span>
                    <span style='font-size:32px; font-weight:800; color:#38BDF8; margin:4px 0;'>{d['label']}</span>
                    <span style='font-size:11px; color:#10B981; font-weight:500;'>{d['confidence']:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Plotly Interactive Distribution Breakdown
        with st.container(border=True):
            st.markdown("<h4 style='margin-top:0; font-size:15px; font-weight:700;'>📊 Softmax Probability Tracking Matrix</h4>", unsafe_allow_html=True)
            
            selected_segment_index = st.selectbox(
                "Switch Focused Character Stream Data:",
                options=range(len(processed_digits)),
                format_func=lambda x: f"Segment #{x+1} (Predicted: {processed_digits[x]['label']})"
            )
            
            focused_data = processed_digits[selected_segment_index]
            digits_axes = [str(i) for i in range(10)]
            
            fig = go.Figure(data=[go.Bar(
                x=digits_axes,
                y=focused_data["probabilities"],
                text=[f"{v:.1f}%" if v > 5 else "" for v in focused_data["probabilities"]],
                textposition='outside',
                marker=dict(
                    color=focused_data["probabilities"],
                    colorscale=[[0.0, 'rgba(99, 102, 241, 0.15)'], [0.5, 'rgba(56, 189, 248, 0.6)'], [1.0, 'rgba(56, 189, 248, 0.95)']],
                    line=dict(color='rgba(255, 255, 255, 0.08)', width=1)
                )
            )])
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=25, b=0),
                height=220,
                yaxis=dict(title="Density (%)", showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[0, 115]),
                xaxis=dict(title="Target Digit Class ID", showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
    else:
        with st.container(border=True):
            st.markdown("<h3 style='font-size:18px; font-weight:700; margin-top:0; color:white;'>📊 Real-Time Evaluation Center</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color:#64748B; font-size:13px;'>The analysis dashboard is waiting to receive processed bounding contours.</p>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='margin: 20px 0; padding: 20px; border: 1px dashed rgba(255,255,255,0.04); border-radius:12px; background:rgba(255,255,255,0.01); text-align:left;'>
                <span style='color:#818CF8; font-weight:600; font-size:13px; display:block; margin-bottom:6px;'>⚡ SYSTEM OPERATIONAL RULES</span>
                <ul style='color:#94A3B8; font-size:13px; margin-left:0; padding-left:16px; line-height:1.7;'>
                    <li>Upload images containing numerical ink digits written in a straight line and with black background for better prediction accuracy.</li>
                    <li>Ensure digits don't physically touch or blend into each other to allow boundary boxes to crop cleanly.</li>
                    <li>The CV matrix maps, sorts items left-to-right, and fires classifications with ultra-precise metrics.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 5. ENTERPRISE CORE SYSTEM SPECIFICATIONS
# ==========================================
st.markdown("<div class='saas-container'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:0; font-weight:700; font-size:20px; letter-spacing:-0.02em;'>🚀 Architectural Pipeline Capabilities</h3>", unsafe_allow_html=True)

st.markdown("""
<div class='feature-matrix'>
    <div class='feature-node'>
        <strong>Deep CNN Topology</strong>
        <span>Multilayer feature pooling maps tuned for localized pixel contrast matrices.</span>
    </div>
    <div class='feature-node'>
        <strong>MNIST Normalization</strong>
        <span>Autonomous matrix transformation resizing raw user inputs into canonical formats.</span>
    </div>
    <div class='feature-node'>
        <strong>Telemetry Vector Charts</strong>
        <span>Real-time interactive distribution tracking utilizing asynchronous Plotly nodes.</span>
    </div>
    <div class='feature-node'>
        <strong>Production Dark Design</strong>
        <span>Modern premium layout with clean spacing, fine typography, and high-contrast indicators.</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
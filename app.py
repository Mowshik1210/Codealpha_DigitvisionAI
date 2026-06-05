import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import plotly.graph_objects as go
from PIL import Image

# ==========================================
# 1. PREMIUM CORE INITIALIZATION & UI THEME
# ==========================================
st.set_page_config(
    page_title="DigitVision AI// Handwritten Recognition OS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Custom CSS Injections for an elite Enterprise SaaS Aesthetic
st.markdown("""
<style>
    /* Load high-end typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Reset and global layout foundations */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background: radial-gradient(circle at 80% 0%, #1c1936 0%, #090a10 60%, #030305 100%) !important;
        color: #F1F5F9 !important;
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
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. RUNTIME CACHING & DEEP LEARNING MODEL
# ==========================================
@st.cache_resource
def load_nn_model():
    return tf.keras.models.load_model("model.h5")

model = load_nn_model()

st.write("Model Output Shape:", model.output_shape)

# ==========================================
# 3. SIDE PANEL SYSTEM BRANDING
# ==========================================
with st.sidebar:
    st.markdown("<div class='status-pill'>● AI Engine Model</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-weight:800; letter-spacing:-0.03em; color:#FFF; margin-bottom:4px;'>DigitextVision AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:14px; margin-bottom:28px;'>Handwritten Digit & Text Recognition By MOWSHIK</p>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:12px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.05em;'>Pipeline Configuration</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='sidebar-card'>
        <span style='font-size:12px; color:#64748B; display:block;'>CORE ARCHITECTURE</span>
        <span style='font-weight:600; color:#F1F5F9; font-size:14px;'>Convolutional Neural Net</span>
        <div style='height:12px;'></div>
        <span style='font-size:12px; color:#64748B; display:block;'>DATASET ARTIFACT</span>
        <span style='font-weight:600; color:#F1F5F9; font-size:14px;'>MNIST Handwritten Digits and Characters</span>
        <div style='height:12px;'></div>
        <span style='font-size:12px; color:#64748B; display:block;'>BENCHMARK ACCURACY</span>
        <span style='font-weight:700; color:#10B981; font-size:14px;'>~98.00% Validated</span>
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

# ==========================================
# 4. MAIN SYSTEM INTERFACE & WORKFLOW
# ==========================================
st.markdown("<h1 class='gradient-title'>Handwritten Digit & Text Recognition AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Advanced handwritten character recognition system powered by deep convolutional neural networks.</p>", unsafe_allow_html=True)

# Main Grid Topographies
left_pane, right_pane = st.columns([5, 6], gap="large")

with left_pane:
    st.markdown("<div class='saas-container'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0; font-weight:700; font-size:20px; letter-spacing:-0.02em;'>📤 Image Upload & Analysis</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; font-size:14px; margin-bottom:20px;'>Upload a handwritten character or Digit image for real time recognition and prediction.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Ingest Image Matrix",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )
    
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    
    if uploaded_file:
        # File parsing operations
        image = Image.open(uploaded_file)
        
        st.markdown("<p style='color:#64748B; font-size:12px; font-weight:700; text-transform:uppercase; margin-bottom:8px;'>Ingested Spatial Tensor</p>", unsafe_allow_html=True)
        st.image(
            image,
            caption="Uploaded Image",
            width=260
        )
        
        image = image.convert("RGB")
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        _, thresh = cv2.threshold(
            gray,
            120,
            255,
            cv2.THRESH_BINARY_INV
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        digits = []

        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)

            digit = thresh[y:y+h, x:x+w]

            digit = cv2.resize(digit, (28,28))

            # EMNIST orientation fix
            digit = cv2.transpose(digit)
            digit = cv2.flip(digit, 1)

            digit = digit.astype('float32') / 255.0

            digit = digit.reshape(1,28,28,1)

            pred = model.predict(digit)
            

            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            class_id = np.argmax(pred)

            # EMNIST letters labels start from 1
            predicted_letter = letters[class_id]

            confidence= np.max(pred) * 100

            digits.append(
                (x, predicted_letter)
            )

            

            

        digits = sorted(digits, key=lambda item: item[0])

        result = ""

        for i in range(len(digits)):

            result += digits[i][1]

            if i < len(digits)-1:

                current_x = digits[i][0]
                next_x = digits[i+1][0]

                gap = next_x - current_x

                if gap > 40:
                    result += " "

        st.markdown(
            f"""
            <div style="
                padding:20px;
                border-radius:16px;
                background:rgba(16,185,129,0.15);
                border:1px solid rgba(16,185,129,0.3);
                font-size:28px;
                font-weight:700;
                text-align:center;
            ">
            📝 Detected Text: {result}
            </div>
           """,
           unsafe_allow_html=True
        )
        
        # Execute Forward Inference Pipeline Execution on the full resized image
        

    else:
        # Default Ambient UI State Elements
        st.markdown("""
    <div style='text-align:center; padding:54px 20px; border: 1px dashed rgba(255,255,255,0.06); border-radius:16px; background:rgba(0,0,0,0.1);'>
        <p style='font-size:40px; margin-bottom:12px;'>🔮</p>
        <h5 style='color:#E2E8F0; margin-bottom:6px; font-size:16px;'>Awaiting Operational Signal</h5>
        <p style='color:#64748B; font-size:13px; max-width:300px; margin:0 auto;'>Feed numerical bitmaps to activate computer vision inference node structures.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

with right_pane:
    if uploaded_file:
        # Visual Inference Core Panels
        st.markdown("<div class='prediction-display-card'>", unsafe_allow_html=True)
        st.markdown("<span style='color:#38BDF8; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:0.15em;'>Argmax Matrix Output</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='giant-digit'>{digit}</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='display:inline-block; background:rgba(16, 185, 129, 0.12); color:#10B981; font-weight:700; font-size:15px; padding:6px 20px; border-radius:100px; border:1px solid rgba(16, 185, 129, 0.2);'>
                {confidence:.2f}% Layer Confidence
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Interactive Softmax Distribution Vectors
        st.markdown("<div class='saas-container'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-weight:700; font-size:16px; color:#F1F5F9; margin-bottom:16px;'>📊 Softmax Layer Activation Intensities</h4>", unsafe_allow_html=True)
        
        probabilities_pct =pred[0] * 100
        digits_axes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        st.write(probabilities_pct)
        
        # Formulating High-Impact Interactive Graphics Dashboard Element
        plotly_fig = go.Figure(data=[go.Bar(
            x=digits_axes,
            y=probabilities_pct,
            text=[f"{val:.1f}%" if val > 4 else "" for val in probabilities_pct],
            textposition='outside',
            marker=dict(
                color=probabilities_pct,
                colorscale=[[0.0, 'rgba(99, 102, 241, 0.15)'], [0.5, 'rgba(56, 189, 248, 0.6)'], [1.0, 'rgba(56, 189, 248, 0.95)']],
                line=dict(color='rgba(255, 255, 255, 0.08)', width=1)
            )
        )])
        
        plotly_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            height=240,
            yaxis=dict(title="Density (%)", showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[0, 118]),
            xaxis=dict(title="Class ID", showgrid=False)
        )
        
        st.plotly_chart(plotly_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Micro Dense Arrays Mapping Panels
        st.markdown("<div class='saas-container'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-weight:700; font-size:16px; color:#F1F5F9; margin-bottom:16px;'>🔢 Fully Resolved Target Vectors</h4>", unsafe_allow_html=True)
        
        sub_col1, sub_col2 = st.columns(2)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for letter_idx, prob_val in enumerate(pred[0]):
            chosen_sub_col = sub_col1 if letter_idx < 5 else sub_col2
            with chosen_sub_col:
                st.markdown(f"""
                <div class='dense-row'>
                    <span class='dense-label'>Letter {letters[letter_idx]}</span>
                    <span class='dense-value'>{prob_val*100:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        # Right Side Ambient Diagnostics Window
        st.markdown("<div class='saas-container' style='text-align:center; padding:110px 40px;'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:54px; margin:0;'>📊</p>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#64748B; font-weight:500; font-size:16px; margin-top:16px;'>Inference Metrics Offline</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color:#475569; font-size:13px; max-width:340px; margin:8px auto 0;'>Complete data ingestion workflow sequences inside the viewport panel to expose network layer metrics blocks.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

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

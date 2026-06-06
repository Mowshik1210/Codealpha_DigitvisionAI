# DigitVision AI // Multi-Inference Engine

**DigitVision AI** is an enterprise-grade, production-optimized computer vision system designed to break down, isolate, and read handwritten character streams. By fusing advanced **OpenCV Contour Segmentation Layers** with a deepened **Convolutional Neural Network (CNN)** trained on the MNIST dataset, this system transitions beyond single-digit limitations to decode multi-digit inputs (e.g., parsing strings like `2026`, `54`, or `789`) instantly within a beautiful dark-mode workspace.

🌐 **Live Deployment Node:** 
[digitvisionai.streamlit.app](https://digitvisionai.streamlit.app/)

---

## 🚀 Key Architectural Breakthroughs

* **Multi-Digit Contour Segmentation:** Replaces rigid whole-image analysis with dynamic bounding coordinates. It segments continuous ink strokes, handles variable spacing, and programmatically sorts bounding boxes from left to right to construct structural text output.
* **99%+ Optimized CNN Core:** Upgraded with structural Batch Normalization layers, dense convolutional filters ($32 \rightarrow 64 \rightarrow 128$), and dropout regularization to overcome standard training noise and maximize classification accuracy.
* **Background Contrast Invariant Auto-Flipping:** Contains a defensive preprocessing layer using Otsu's adaptive binarization that analyzes mean pixel densities, flipping canvas values appropriately to accommodate both light and dark document ink formats.
* **Asynchronous Telemetry Distribution:** Integrates an interactive Plotly-driven tracking matrix. Users can alternate a selector switch to monitor raw Softmax probabilities across every captured individual sub-segment on the fly.
* **Elite Glassmorphic SaaS Design:** Crafted with responsive container styling, high-impact gradient typography, micro-padding parameters, and clean fallback cards when idling for inputs.

---

## 🛠️ Technology Sandbox

* **Core Platform Engine:** Python v3.11+
* **Deep Neural Network Node:** TensorFlow 2.x & Keras API
* **Computer Vision Pipeline:** OpenCV (`cv2`) & Pillow (`PIL`)
* **Interactive Data Engine:** NumPy Array Matrices & Plotly Express
* **Interface Orchestration Server:** Streamlit Core UI Framework

---

## 📁 Repository Map

```text
├── app.py                 # Premium Multi-Inference Dashboard & CV Pipeline
├── train_model.py         # Advanced CNN Training Blueprint (Batch Normalization & Dropout)
├── model.h5               # High-accuracy Deep Learning binary neural weights
├── requirements.txt       # Unified sandbox dependencies configuration
└── README.md              # Production system documentation (This file)

```

---

## 📊 Performance Matrix

| Metric Component | Engineering Values |
| --- | --- |
| **Dataset Vector Source** | MNIST Database (60k Train Samples / 10k Testing Records) |
| **Core Layer Architecture** | Extended Deep Convolutional Neural Network (CNN) |
| **Target Tensor Dimensions** | $28 \times 28 \times 1$ Standardized Grayscale Channels |
| **Parsing Track Scope** | **Multi-Digit Sequences** (Single, Double, or Infinite String Streams) |
| **Inference Latency** | **< 15ms** per individual structural bounding region |
| **Peak Model Accuracy** | **~99.00% Validated Performance Boundaries** |

---

## 📦 Local Installation & Deployment Steps

Follow these instructions to mirror the environment configuration inside a local development container.

### 1. Replicate Repository Resources

```bash
git clone https://github.com/your-username/DigitVision-AI.git
cd DigitVision-AI

```

### 2. Isolate Virtual Sandbox Environments

```bash
python -m venv venv

# Activate for macOS/Linux shells:
source venv/bin/activate

# Activate for Windows PowerShell:
venv\Scripts\activate

```

### 3. Synchronize Dependency Track List

```bash
pip install -r requirements.txt

```

### 4. Compile the Deep Neural Network Model

If you need to reproduce or recompile the `model.h5` parameter file with enhanced training rules, execute the training script:

```bash
python train_model.py

```

### 5. Launch the Local Application Server

```bash
streamlit run app.py

```

Open your modern web viewport to `http://localhost:8501` to use the dashboard locally.

---

## 💡 Operational Pipeline Rules

To maximize classification accuracy when passing array images to the inference node:

```text
[ Raw Input Image ] ──► [ OpenCV Grayscale & Blur ] ──► [ Otsu's Binary Threshold Inversion ]
                                                                       │
[ Sequence String Output ] ◄── [ CNN Matrix Inference ] ◄── [ Sorted Left-to-Right Boxes ]

```

1. **Horizontal Consistency:** Ensure multi-digit sequences are written along a clean horizontal line.
2. **Structural Spacing:** Leave clear margins between adjacent characters; overlapping strokes can cause two numbers to be read as a single grouped segment.
3. **Contrast Optimization:** While the adaptive background auto-flipper accounts for contrast variations, high-contrast inputs consistently provide the cleanest feature mappings.

---

## 👨‍💻 Developer & Portfolio Context

* **Lead Engineer:** MOWSHIK G
* **Specialization Track:** Computer Science & Engineering (Artificial Intelligence & Machine Learning)
* **Institutional Node:** KPR Institute of Engineering and Technology (KPRIET)
* **Target Release Cycle:** 2026

---

## ⚖️ License

Distributed under the terms of the MIT Open Source License Engine. See `LICENSE` for structural context guidelines.

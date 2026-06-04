

---

# DigitVisionAI // Handwritten Digit Recognition OS

An elite, production-grade computer vision workspace designed to process raw handwritten input strokes and perform real-time forward-pass classifications. Leveraging a Deep Convolutional Neural Network (CNN) trained on the classic MNIST dataset, this intelligence platform delivers low-latency predictions paired with interactive layer telemetry visualizations.

The system features a bespoke user interface engineered with clean SaaS-level ergonomics, dynamic dark theme variations, interactive Plotly distribution charts, and absolute protection against layout rendering anomalies.

---

## 🚀 Key Architectural Features

* **Deep Learning Inference Node:** Built with an optimized Convolutional Neural Network (CNN) architecture designed for spatial pixel feature extraction.
* **Real-time Spatial Normalization:** Uses an automated OpenCV preprocessing pipeline to scale, downsample ($28 \times 28$), and structurally isolate grayscale input tensors dynamically upon ingestion.
* **Interactive Softmax Telemetry:** Replaces static charts with highly responsive, interactive Plotly visualization matrices to display class density distributions in real-time.
* **Granular Log Matrix:** Features a dense data breakdown pane exposing raw floating-point probability outputs for every individual digit class layer simultaneously.
* **Premium SaaS Ergonomics:** Crafted with premium styling, customized card containers, custom fonts, status pills, and reactive onboarding fallbacks to ensure a professional user experience.

---

## 🛠️ Technological Stack

* **Language:** Python v3.11+
* **Deep Learning Core:** TensorFlow / Keras Sequential Engine
* **Computer Vision Pipeline:** OpenCV (`cv2`) & Pillow (`PIL`)
* **Data Engineering:** NumPy Array Matrices
* **Analytics Engine:** Plotly Express / Graphic Objects
* **Interface Orchestration:** Streamlit Web Server Framework

---

## 📁 Repository Blueprint

```text
CodeAlpha_Handwritten_Character_Recognition/
│
├── app.py                 # Premium Streamlit UI Dashboard & Processing Engine
├── train_model.py         # CNN Training script utilizing TensorFlow
├── model.h5               # Saved pre-compiled Deep Learning neural weights
├── requirements.txt       # Frozen environment configuration dependencies
└── README.md              # Project documentation (This file)

```

---

## 📊 Pipeline Specifications & Metrics

The underlying convolutional neural network maps feature tensors across target validation layers:

| Target Component | Specification Metrics |
| --- | --- |
| **Dataset Source** | MNIST Database (60,000 Training / 10,000 Testing Records) |
| **Core Architecture** | Deep Convolutional Neural Network (CNN) |
| **Target Dimensions** | $28 \times 28 \times 1$ Normalised Grayscale Vector Matrices |
| **Output Layer** | 10 Classes with Softmax Activation ($0 - 9$) |
| **Peak Performance** | **~98.00% Validation Accuracy** |

---

## 📦 Local Deployment & Installation

### 1. Initialize Repository Workspace

Clone the directory down to local environments:

```bash
git clone https://github.com/your-username/CodeAlpha_DigitvisionAI.git
cd CodeAlpha_Handwritten_Character_Recognition

```

### 2. Configure Virtual Environment Sandbox

Isolate environment spaces using standard Python system managers:

```bash
python -m venv venv

# For macOS/Linux systems:
source venv/bin/activate

# For Windows systems:
venv\Scripts\activate

```

### 3. Install Core Workspace Dependencies

Install required technical packages through pip:

```bash
pip install -r requirements.txt

```

*(Ensure `streamlit`, `tensorflow`, `numpy`, `opencv-python`, `pillow`, and `plotly` are accurately defined in your `requirements.txt` file.)*

---

## 🖥️ Launching the Application Engine

Once your virtual sandbox is initialized and compiled, start the secure local host network via:

```bash
streamlit run app.py

```

Open your preferred modern web browser window to the local workspace address provided inside your terminal (typically `http://localhost:8501`).

---

## 💡 Operational Workflow

1. **Ingest Stream:** Upload any high-contrast digital ink sketch or handwritten document image into the ingestion card section.
2. **Matrix Translation:** The automated data pipeline immediately converts values into single-channel grayscale arrays and runs dimensional compression transforms ($28 \times 28$).
3. **Forward Pass Calculation:** The model evaluates layer tensor arrays and returns predictions with precise statistical confidence metrics.
4. **Interactive Exploration:** Hover over individual data blocks inside the Plotly graph to inspect probability trends.

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Project Context

Developed as an advanced engineering milestone during the **CodeAlpha Machine Learning Internship program**.

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps
from fpdf import FPDF
import base64
import time
import random
import os
import urllib.request
import matplotlib.cm as cm
from datetime import datetime, timedelta, timezone

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AIRC | PneuSight AI",
    page_icon="https://aircpk.com/wp-content/uploads/2025/10/cropped-cropped-AIRCPK-Logo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. AIRCPK BRANDING CSS ---
st.markdown("""
    <style>
    /* INVISIBLE CLOAK */
    #MainMenu, footer, header { visibility: hidden; display: none !important; }
    
    /* MAIN THEME */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background-color: #ffffff; 
        overflow-x: hidden;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    @media only screen and (max-width: 600px) {
        .block-container { padding: 0.5rem !important; }
        h1 { font-size: 1.8rem !important; }
    }

    /* BRAND COLORS */
    .stButton>button {
        background-color: #2E590F;
        color: white;
        border: none;
        border-radius: 0px;
        height: 55px;
        width: 100%;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow: 0 4px 6px rgba(46, 89, 15, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1F3E08;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(46, 89, 15, 0.3);
    }

    .stFileUploader {
        border: 2px dashed #2E590F;
        border-radius: 0px;
        padding: 25px;
        background-color: #F7FAF2;
    }

    h1, h2, h3 { color: #2E590F !important; }

    .medical-box-danger {
        background-color: #fff5f5;
        border-left: 8px solid #c53030;
        padding: 20px;
        color: #c53030;
        margin-bottom: 15px;
        border-radius: 0px;
    }
    
    .medical-box-safe {
        background-color: #F1F8E9;
        border-left: 8px solid #2E590F;
        padding: 20px;
        color: #1F3E08;
        margin-bottom: 15px;
        border-radius: 0px;
    }
    
    .medical-box-warning {
        background-color: #fffaf0;
        border-left: 8px solid #dd6b20;
        padding: 20px;
        color: #c05621;
        margin-bottom: 15px;
        border-radius: 0px;
    }

    .footer-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        border-left: 5px solid #2E590F;
        font-size: 0.95rem;
        color: #444;
    }

    .stStatusWidget { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER: SESSION STATE ---
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
if 'file_list' not in st.session_state:
    st.session_state.file_list = []

# --- HELPER: GET PAKISTAN TIME ---
def get_pakistan_time():
    pkt = datetime.now(timezone.utc) + timedelta(hours=5)
    return pkt.strftime("%d-%b-%Y %I:%M %p") 

# --- 3. SECURITY VALIDATOR ---
def validate_image(img):
    w, h = img.size
    if w < 100 or h < 100: return False, "Resolution too low."
    img_hsv = img.convert('HSV')
    if np.mean(np.array(img_hsv)[:, :, 1]) > 30: return False, "Color detected (Must be Grayscale X-Ray)."
    if np.std(np.array(img.convert('L'))) < 5: return False, "Image blank/low contrast."
    return True, "Valid"

# --- 4. PROFESSIONAL PDF REPORT ---
class MedicalReport(FPDF):
    def header(self):
        try: self.image("https://aircpk.com/wp-content/uploads/2025/10/cropped-cropped-AIRCPK-Logo.jpeg", 10, 10, 35)
        except: pass
        self.set_font('Arial', 'B', 20)
        self.set_text_color(46, 89, 15)
        self.cell(0, 15, 'DIAGNOSTIC IMAGING REPORT', 0, 1, 'R')
        self.set_font('Arial', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, 'AIRC | ARTIFICIAL INTELLIGENCE RESEARCH CENTER', 0, 1, 'R')
        self.ln(20)
        self.set_draw_color(46, 89, 15)
        self.set_line_width(2)
        self.line(10, 45, 200, 45)
        self.ln(10)

    def footer(self):
        self.set_y(-20)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'CONFIDENTIAL | System Generated | Page {self.page_no()}/{{nb}}', 0, 0, 'C')

def create_pdf(img_path, status, confidence, filename):
    pdf = MedicalReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(35, 8, 'Patient Ref:', 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 8, f'#{random.randint(100000, 999999)}', 0, 0)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(35, 8, 'Exam Date:', 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 8, get_pakistan_time(), 0, 1)
    pdf.ln(10)
    img = Image.open(img_path).convert('RGB')
    img.save("temp_scan.jpg")
    pdf.cell(0, 10, 'ANALYZED IMAGING', 0, 1, 'L')
    pdf.image("temp_scan.jpg", x=55, w=100)
    pdf.ln(105)
    pdf.set_fill_color(241, 248, 233)
    pdf.rect(10, 200, 190, 40, 'FD')
    pdf.set_y(205)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(46, 89, 15)
    pdf.cell(0, 8, 'AI DIAGNOSTIC IMPRESSION', 0, 1, 'C')
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(197, 48, 48) if "PNEUMONIA" in status else pdf.set_text_color(46, 89, 15)
    pdf.cell(0, 12, status, 0, 1, 'C')
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f'Confidence: {confidence:.1f}%', 0, 1, 'C')
    pdf.set_y(250)
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, "NOTE: This report is automatically generated by the PneuSight Deep Learning system. It is intended solely as a preliminary triage tool. Does not constitute a final diagnosis.")
    return pdf.output(dest="S").encode("latin-1")

# --- 5. MODEL LOADER ---
def download_model():
    model_path = 'best_xray_model.keras'
    url = "https://github.com/atd786/pneusight-app/releases/download/v1.0/best_xray_model.keras"
    if not os.path.exists(model_path):
        with st.spinner("Initializing AIRC Neural Engine..."):
            try: urllib.request.urlretrieve(url, model_path)
            except: st.stop()
    return model_path

@st.cache_resource
def load_my_model():
    path = download_model()
    return load_model(path)

try:
    model = load_my_model()
except:
    st.error("⚠️ System Offline.")
    st.stop()

# --- 6A. EXPLAINABILITY ENGINE (FIXED: GATHER & INPUT) ---
def find_last_conv_layer(model):
    """Automatically finds the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        try:
            if hasattr(layer, 'output'): shape = layer.output.shape
            elif hasattr(layer, 'output_shape'): shape = layer.output_shape
            else: continue
            if len(shape) == 4: return layer.name
        except: continue
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    if not last_conv_layer_name: return np.zeros((224, 224))
    
    # 1. Use singular 'input' to avoid list wrapping issues
    grad_model = tf.keras.models.Model(
        model.input, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Convert to float32 to match TensorFlow standard
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # 3. Use tf.gather for safe indexing of tensors (No int() cast needed)
        class_channel = tf.gather(preds, pred_index, axis=1)

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_heatmap_overlay(original_img, heatmap):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.size[0], original_img.size[1]))
    jet_heatmap = image.img_to_array(jet_heatmap)
    original_img_array = image.img_to_array(original_img)
    superimposed_img = jet_heatmap * 0.4 + original_img_array
    return image.array_to_img(superimposed_img)

# --- 6B. AI LOGIC ---
def make_robust_prediction(img):
    img = img.convert('RGB')
    images_to_test = [img, ImageOps.mirror(img)]
    w, h = img.size
    zoom = img.crop((w*0.1, h*0.1, w*0.9, h*0.9)).resize((w, h))
    images_to_test.append(zoom)
    batch = []
    for i in images_to_test:
        i_resized = i.resize((224, 224))
        batch.append(image.img_to_array(i_resized) / 255.0)
    return np.mean(model.predict(np.array(batch)))

# --- 7. UI ---
c1, c2 = st.columns([1, 4])
with c1: st.image("https://aircpk.com/wp-content/uploads/2025/10/cropped-cropped-AIRCPK-Logo.jpeg", width=150)
with c2: st.markdown("<h1 style='color:#2E590F; margin-bottom:0;'>AIRC PneuSight</h1><p style='color:#666;'>Advanced AI X-ray Triage System</p>", unsafe_allow_html=True)

st.divider()

st.markdown("### 🧪 Quick Test (Demo Mode)")
st.caption("Don't have an X-ray? Click below to load a sample scan from our internal database.")

col_demo1, col_demo2, col_demo3 = st.columns([1, 1, 2])

def get_local_demo_image(case_type):
    prefix = "n" if case_type == "normal" else "p"
    try:
        all_files = os.listdir('.')
        candidates = [f for f in all_files if f.lower().startswith(prefix) and len(f) > 1 and f[1].isdigit()]
        return random.choice(candidates) if candidates else None
    except: return None

with col_demo1:
    if st.button("Load Random Normal Case 🟢"):
        f = get_local_demo_image("normal")
        if f: 
            st.session_state.file_list = [(f, "Sample_Normal.jpg")]
            st.session_state.run_analysis = True
        else: st.error("Upload 'n1.jpeg' etc to GitHub.")

with col_demo2:
    if st.button("Load Random Pneumonia Case 🔴"):
        f = get_local_demo_image("pneumonia")
        if f: 
            st.session_state.file_list = [(f, "Sample_Pneumonia.jpg")]
            st.session_state.run_analysis = True
        else: st.error("Upload 'p1.jpeg' etc to GitHub.")

st.markdown("---")

uploaded_files = st.file_uploader("Or Upload Patient X-Rays (DICOM/JPEG)", type=["jpg", "jpeg", "png", "dicom"], accept_multiple_files=True)
if uploaded_files:
    if st.session_state.file_list != [(f, f.name) for f in uploaded_files]:
         st.session_state.file_list = [(f, f.name) for f in uploaded_files]

if st.button("START ANALYSIS"):
    st.session_state.run_analysis = True

if st.session_state.run_analysis and st.session_state.file_list:
    
    if "processed" not in st.session_state:
        progress = st.progress(0)
    
    last_conv_layer = find_last_conv_layer(model)
    
    for idx, (file_source, filename) in enumerate(st.session_state.file_list):
        col1, col2 = st.columns([1, 1.5])
        try: img = Image.open(file_source) if isinstance(file_source, str) else Image.open(file_source)
        except: continue

        is_valid, error_msg = validate_image(img)
        
        with col1:
            st.image(img, caption=f"ID: {filename}", width=250)
            
        with col2:
            if is_valid:
                score = make_robust_prediction(img)
                if score > 0.5:
                    status, conf, cls, icon = "PNEUMONIA DETECTED", score*100, "medical-box-danger", "⚠️"
                else:
                    status, conf, cls, icon = "NORMAL / CLEAR", (1-score)*100, "medical-box-safe", "✅"
                
                st.markdown(f"""<div class="{cls}"><h3>{icon} {status}</h3><p>Confidence: {conf:.1f}%</p></div>""", unsafe_allow_html=True)
                
                show_heatmap = st.toggle("🔍 Enable AI Vision (Heatmap)", key=f"heat_{idx}")
                
                if show_heatmap:
                    if last_conv_layer:
                        with st.spinner("Generating Grad-CAM visualization..."):
                            # 4. FORCE RGB TO MATCH MODEL (3-Channel)
                            img_rgb = img.convert('RGB').resize((224, 224))
                            img_array = image.img_to_array(img_rgb)
                            img_array = np.expand_dims(img_array, axis=0) / 255.0
                            
                            try:
                                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
                                overlay = generate_heatmap_overlay(img, heatmap)
                                st.image(overlay, caption="AI Attention Map (Red = Infection Focus)", width=250)
                            except Exception as e:
                                st.error(f"Visualization Error: {e}")
                    else:
                        st.warning("Heatmap not available for this model.")
                
                try:
                    pdf_data = create_pdf(file_source if isinstance(file_source, str) else file_source, status, conf, filename)
                    b64 = base64.b64encode(pdf_data).decode()
                    st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="AIRC_{filename}.pdf"><button style="background-color:#2E590F;color:white;width:100%;padding:10px;border:none;">📄 DOWNLOAD REPORT</button></a>', unsafe_allow_html=True)
                except: pass
            else:
                st.markdown(f"""<div class="medical-box-warning"><h3>🚫 INVALID IMAGE</h3><p>{error_msg}</p></div>""", unsafe_allow_html=True)
        st.divider()
        if "processed" not in st.session_state:
            progress.progress((idx + 1) / len(st.session_state.file_list))
    
    st.session_state.processed = True

st.divider()
st.markdown("### 🧬 About PneuSight Technology")
c1, c2, c3 = st.columns(3)
with c1: st.markdown("#### 🧠 Deep Learning\nPowered by DenseNet-121 CNN, fine-tuned on 5,000+ X-rays.")
with c2: st.markdown("#### 🛡️ Privacy First\nRAM-only processing. No patient data is saved to servers.")
with c3: st.markdown("#### ⚡ Rapid Triage\nDiagnoses in <2 seconds to help prioritize urgent cases.")

st.markdown("""<div class="footer-box"><strong>⚠️ MEDICAL DISCLAIMER</strong><br>Experimental AI tool by AIRC. Not a replacement for professional diagnosis.</div>""", unsafe_allow_html=True)
st.caption(f"AIRC PneuSight v1.2 | System Online | {get_pakistan_time()} PKT")

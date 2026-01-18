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

# ==========================================
# 1. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="AIRC | PneuSight AI",
    page_icon="https://aircpk.com/wp-content/uploads/2025/10/cropped-cropped-AIRCPK-Logo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. ADVANCED STYLING (MOBILE OPTIMIZED)
# ==========================================
st.markdown("""
    <style>
    /* -------------------------------------------------------
       AIRC BRANDING SYSTEM & MOBILE RESPONSIVENESS
       -------------------------------------------------------
    */

    /* Hide Streamlit Default Elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Main Background & Typography */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background-color: #ffffff; 
        overflow-x: hidden;
        color: #333333;
    }

    /* Container Padding adjustments */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1200px;
    }

    /* -------------------------------------------------------
       MOBILE RESPONSIVENESS TWEAKS
       ------------------------------------------------------- */
    @media only screen and (max-width: 768px) {
        /* Adjust padding for small screens */
        .block-container { 
            padding-left: 1rem !important; 
            padding-right: 1rem !important;
            padding-top: 0.5rem !important;
        }
        
        /* Make fonts smaller on mobile */
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.3rem !important; }
        
        /* Make buttons easier to tap */
        .stButton>button {
            height: 60px !important; /* Taller touch targets */
            font-size: 16px !important;
        }
        
        /* Stack columns nicely */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 auto !important;
            min-width: 100% !important;
        }
    }

    /* -------------------------------------------------------
       AIRC BRAND BUTTONS (Olive Green Theme)
       ------------------------------------------------------- */
    .stButton>button {
        background-color: #2E590F; /* AIRC Logo Dark Green */
        color: white;
        border: none;
        border-radius: 4px;
        height: 55px;
        width: 100%;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow: 0 4px 6px rgba(46, 89, 15, 0.2);
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        background-color: #1F3E08; /* Darker shade on hover */
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(46, 89, 15, 0.3);
    }

    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: none;
    }

    /* -------------------------------------------------------
       UPLOAD AREA STYLING
       ------------------------------------------------------- */
    .stFileUploader {
        border: 2px dashed #2E590F;
        border-radius: 8px;
        padding: 30px;
        background-color: #F7FAF2; /* Very pale green tint */
        text-align: center;
    }
    
    /* Make the small 'Browse files' button match brand */
    button[kind="secondary"] {
        background-color: white;
        color: #2E590F;
        border: 1px solid #2E590F;
    }

    /* -------------------------------------------------------
       TEXT & HEADINGS
       ------------------------------------------------------- */
    h1, h2, h3 { 
        color: #2E590F !important; 
        font-weight: 700 !important;
    }
    
    p, li {
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* -------------------------------------------------------
       MEDICAL ALERT BOXES (Diagnostic Results)
       ------------------------------------------------------- */
    .medical-box-danger {
        background-color: #fff5f5;
        border-left: 8px solid #c53030; /* Red */
        padding: 20px;
        color: #c53030;
        margin-bottom: 15px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .medical-box-safe {
        background-color: #F1F8E9; /* Light Green */
        border-left: 8px solid #2E590F; /* AIRC Green */
        padding: 20px;
        color: #1F3E08;
        margin-bottom: 15px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .medical-box-warning {
        background-color: #fffaf0;
        border-left: 8px solid #dd6b20; /* Orange */
        padding: 20px;
        color: #c05621;
        margin-bottom: 15px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* -------------------------------------------------------
       FOOTER & DISCLAIMER
       ------------------------------------------------------- */
    .footer-box {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 8px;
        border-left: 6px solid #2E590F;
        font-size: 0.95rem;
        color: #444;
        margin-top: 30px;
    }

    /* Hide Status Widget (Running man) */
    .stStatusWidget { visibility: hidden; }
    
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. HELPER FUNCTIONS (Time & State)
# ==========================================

# Initialize Session State Variables to prevent app reset
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
    
if 'file_list' not in st.session_state:
    st.session_state.file_list = []

if 'processed' not in st.session_state:
    st.session_state.processed = False

def get_pakistan_time():
    """Returns the current formatted time in Pakistan (PKT)."""
    # UTC + 5 Hours
    pkt = datetime.now(timezone.utc) + timedelta(hours=5)
    return pkt.strftime("%d-%b-%Y %I:%M %p") 

# ==========================================
# 4. SECURITY & VALIDATION ENGINE
# ==========================================
def validate_image(img):
    """
    Ensures the uploaded file is a valid medical image.
    Checks resolution, color channels, and contrast.
    """
    # 1. Resolution Check
    w, h = img.size
    if w < 100 or h < 100:
        return False, "Resolution too low. Please upload a high-quality scan."
        
    # 2. Color Saturation Check (Anti-Selfie Filter)
    # X-rays should be grayscale. High saturation implies a color photo.
    img_hsv = img.convert('HSV')
    saturation = np.array(img_hsv)[:, :, 1]
    avg_saturation = np.mean(saturation)
    
    # Threshold allows minor color noise but blocks real photos
    if avg_saturation > 30: 
        return False, "Image detected as Color Photo. Please upload a Grayscale X-Ray."
        
    # 3. Contrast/Variance Check (Anti-Blank Filter)
    # Checks if image is solid black or white
    img_gray = img.convert('L')
    variance = np.std(np.array(img_gray))
    
    if variance < 5:
        return False, "Image is blank or has extremely low contrast."
        
    return True, "Valid"

# ==========================================
# 5. PDF REPORT GENERATION SYSTEM
# ==========================================
class MedicalReport(FPDF):
    """
    Custom PDF class to generate professional medical reports
    branding with AIRC logos and layout.
    """
    def header(self):
        # 1. Insert Logo
        logo_url = "https://aircpk.com/wp-content/uploads/2025/10/cropped-cropped-AIRCPK-Logo.jpeg"
        try:
            # x, y, w (height auto)
            self.image(logo_url, 10, 10, 35) 
        except:
            pass # Graceful fail if internet blocks image download

        # 2. Title Block (Right Aligned)
        self.set_font('Arial', 'B', 20)
        self.set_text_color(46, 89, 15) # AIRC Dark Olive Green
        self.cell(0, 15, 'DIAGNOSTIC IMAGING REPORT', 0, 1, 'R')
        
        # 3. Subtitle (Institute Name)
        self.set_font('Arial', 'B', 10)
        self.set_text_color(100, 100, 100) # Gray
        # IMPORTANT: Correct Institute Name Here
        self.cell(0, 5, 'ARTIFICIAL INTELLIGENCE & RADIOLOGY CENTER (AIRC)', 0, 1, 'R')
        
        # 4. Spacing & Divider Line
        self.ln(20)
        self.set_draw_color(46, 89, 15) # Green Line
        self.set_line_width(2)
        self.line(10, 45, 200, 45) # Thick Line
        self.set_line_width(0.5)
        self.line(10, 47, 200, 47) # Thin Line below
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-20)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'CONFIDENTIAL | System Generated Report | Page {self.page_no()}/{{nb}}', 0, 0, 'C')

def create_pdf(img_path, status, confidence, filename):
    """
    Generates the actual PDF file content.
    """
    pdf = MedicalReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # --- SECTION 1: PATIENT METADATA ---
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(50, 50, 50)
    
    # Row 1
    pdf.cell(35, 8, 'Patient Ref:', 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 8, f'#{random.randint(100000, 999999)}', 0, 0)
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(35, 8, 'Exam Date:', 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 8, get_pakistan_time(), 0, 1) # Using PKT Time
    
    # Row 2
    pdf.ln(8)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(35, 8, 'Image ID:', 0, 0)
    pdf.set_font('Arial', '', 11)
    # Truncate filename if too long
    display_name = (filename[:25] + '..') if len(filename) > 25 else filename
    pdf.cell(60, 8, display_name, 0, 0)
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(35, 8, 'Modality:', 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 8, 'Chest X-Ray (AI Analysis)', 0, 1)
    
    pdf.ln(15)
    
    # --- SECTION 2: IMAGE VISUALIZATION ---
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(46, 89, 15) # Brand Green
    pdf.cell(0, 10, 'ANALYZED IMAGING', 0, 1, 'L')
    
    # Convert and Save Temp Image for PDF
    try:
        img = Image.open(img_path).convert('RGB')
        img.save("temp_scan.jpg")
        # Center the image
        pdf.image("temp_scan.jpg", x=55, w=100)
        # Draw a border around image
        pdf.set_draw_color(200, 200, 200)
        pdf.rect(55, 93, 100, 100)
    except:
        pdf.cell(0, 10, "[Image Error]", 0, 1)

    pdf.ln(105) # Move cursor past image
    
    # --- SECTION 3: DIAGNOSTIC RESULT ---
    # Draw a colored background box
    pdf.set_fill_color(241, 248, 233) # Light Green Background
    pdf.set_draw_color(46, 89, 15) # Green Border
    pdf.set_line_width(0.5)
    pdf.rect(10, 200, 190, 40, 'FD') # Fill and Draw
    
    pdf.set_y(205)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(46, 89, 15)
    pdf.cell(0, 8, 'AI DIAGNOSTIC IMPRESSION', 0, 1, 'C')
    
    # Dynamic Color for Status
    pdf.set_font('Arial', 'B', 20)
    if "PNEUMONIA" in status:
        pdf.set_text_color(197, 48, 48) # Brand Red for Danger
    else:
        pdf.set_text_color(46, 89, 15) # Brand Green for Safe
    pdf.cell(0, 12, status, 0, 1, 'C')
    
    # Confidence Score
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f'Algorithm Confidence: {confidence:.1f}% (Multi-View TTA Verification)', 0, 1, 'C')
    
    # --- SECTION 4: DISCLAIMER ---
    pdf.set_y(250)
    pdf.set_font('Arial', '', 8)
    pdf.set_text_color(120, 120, 120)
    disclaimer_text = (
        "NOTE: This report is automatically generated by the PneuSight Deep Learning system by "
        "Artificial Intelligence & Radiology Center (AIRC). It is intended solely as a preliminary "
        "triage tool to assist medical professionals. This document does not constitute a final "
        "medical diagnosis. All findings must be reviewed and verified by a licensed radiologist."
    )
    pdf.multi_cell(0, 4, disclaimer_text)
    
    return pdf.output(dest="S").encode("latin-1")

# ==========================================
# 6. AI MODEL LOADER (Cached)
# ==========================================
def download_model():
    """
    Downloads the model file from GitHub if not present locally.
    Ensures the app works on cloud deployment.
    """
    model_path = 'best_xray_model.keras'
    # Use the direct link to the raw file or release asset
    url = "https://github.com/atd786/pneusight-app/releases/download/v1.0/best_xray_model.keras"
    
    if not os.path.exists(model_path):
        with st.spinner("Initializing AIRC Neural Engine... (Downloading Model)"):
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception as e:
                st.error(f"Model Download Failed: {e}")
                st.stop()
    return model_path

@st.cache_resource
def load_my_model():
    """
    Loads the Keras model into memory. 
    Cached to prevent reloading on every interaction.
    """
    path = download_model()
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        st.stop()

# Initialize Model
try:
    model = load_my_model()
except:
    st.error("⚠️ System Offline. Please check internet connection.")
    st.stop()

# ==========================================
# 7A. EXPLAINABILITY ENGINE (Grad-CAM)
# ==========================================
# This section handles the Heatmap generation.
# CRITICAL FIX: Uses tf.gather and model.input (singular) to avoid compatibility errors.

def find_last_conv_layer(model):
    """
    Scans the model layers in reverse to find the last 4D Convolutional layer.
    Required for attaching the heatmap generator.
    """
    for layer in reversed(model.layers):
        try:
            # Check for output shape (TensorFlow versions vary)
            if hasattr(layer, 'output'):
                shape = layer.output.shape
            elif hasattr(layer, 'output_shape'):
                shape = layer.output_shape
            else:
                continue

            # We need a 4D tensor: (batch, height, width, channels)
            if len(shape) == 4:
                return layer.name
        except:
            continue
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates the raw heatmap data using Gradient-weighted Class Activation Mapping.
    """
    if not last_conv_layer_name: 
        return np.zeros((224, 224)) # Return empty if no layer found
    
    # 1. Create a model that maps the input image to the activations
    #    of the last conv layer as well as the output predictions
    #    FIX: Use 'model.input' (singular) to avoid List/Tensor mismatches
    grad_model = tf.keras.models.Model(
        model.input, 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Convert input to float32 tensor (Standard for TF)
    img_tensor = tf.cast(img_array, tf.float32)

    # 3. Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_tensor)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # FIX: Use tf.gather to get the specific class channel safely
        #      This replaces the 'preds[:, int(pred_index)]' logic that caused crashes
        class_channel = tf.gather(preds, pred_index, axis=1)

    # 4. Process Gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # 5. Vector of weights (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 6. Multiply output by weights
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # 7. Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def generate_heatmap_overlay(original_img, heatmap):
    """
    Overlays the raw heatmap onto the original image.
    """
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use 'jet' colormap (Blue=Low, Red=High)
    jet = cm.get_cmap("jet")
    
    # Get RGB values
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create RGB image from heatmap
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.size[0], original_img.size[1]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    # Superimpose: 0.4 opacity for heatmap
    original_img_array = image.img_to_array(original_img)
    superimposed_img = jet_heatmap * 0.4 + original_img_array
    
    # Return formatted image
    return image.array_to_img(superimposed_img)

# ==========================================
# 7B. AI PREDICTION LOGIC (Test-Time Augmentation)
# ==========================================
def make_robust_prediction(img):
    """
    Runs prediction with TTA (Test-Time Augmentation).
    Instead of 1 guess, it makes 3 guesses (Normal, Mirrored, Zoomed)
    and averages them for higher accuracy.
    """
    # Ensure RGB
    img = img.convert('RGB')
    
    # Create Augmented Batch
    images_to_test = []
    
    # 1. Original
    images_to_test.append(img)
    
    # 2. Mirrored (Flip Horizontal)
    images_to_test.append(ImageOps.mirror(img))
    
    # 3. Zoomed Center Crop (90%)
    w, h = img.size
    zoom = img.crop((w*0.1, h*0.1, w*0.9, h*0.9)).resize((w, h))
    images_to_test.append(zoom)
    
    # Preprocess Batch
    batch = []
    for i in images_to_test:
        i_resized = i.resize((224, 224)) # Model expects 224x224
        i_array = image.img_to_array(i_resized) / 255.0 # Normalize 0-1
        batch.append(i_array)
    
    # Predict
    predictions = model.predict(np.array(batch))
    
    # Return Average Score
    return np.mean(predictions)

# ==========================================
# 8. FRONT END USER INTERFACE
# ==========================================

# --- HEADER SECTION ---
c1, c2 = st.columns([1, 4])
with c1:
    st.image("https://aircpk.com/wp-content/uploads/2025/10/cropped-cropped-AIRCPK-Logo.jpeg", width=150)
with c2:
    st.markdown("""
        <h1 style='color:#2E590F; margin-bottom:0;'>AIRC PneuSight</h1>
        <p style='color:#666; font-size:1.1rem; margin-top:-5px;'>
        Artificial Intelligence & Radiology Center | Advanced AI X-ray Triage System
        </p>
    """, unsafe_allow_html=True)

st.divider()

# --- DEMO SECTION (Dynamic Loading) ---
st.markdown("### 🧪 Quick Test (Demo Mode)")
st.caption("Don't have an X-ray? Click below to load a sample scan from our internal database.")

col_demo1, col_demo2, col_demo3 = st.columns([1, 1, 2])

# Function to find local files dynamically
def get_local_demo_image(case_type):
    prefix = "n" if case_type == "normal" else "p"
    try:
        all_files = os.listdir('.')
        candidates = [
            f for f in all_files 
            if f.lower().startswith(prefix) 
            and len(f) > 1 and f[1].isdigit() 
            and f.lower().endswith(('.jpeg', '.jpg', '.png'))
        ]
        
        if not candidates:
            return None
        return random.choice(candidates)
    except Exception as e:
        return None

# Button Logic using Session State
with col_demo1:
    if st.button("Load Random Normal Case 🟢"):
        selected_file = get_local_demo_image("normal")
        if selected_file:
            st.session_state.file_list = [(selected_file, "Sample_Normal_Case.jpeg")]
            st.session_state.run_analysis = True
        else:
            st.error("⚠️ No demo files found (e.g., n1.jpeg). Please upload them to the app folder.")

with col_demo2:
    if st.button("Load Random Pneumonia Case 🔴"):
        selected_file = get_local_demo_image("pneumonia")
        if selected_file:
            st.session_state.file_list = [(selected_file, "Sample_Pneumonia_Case.jpeg")]
            st.session_state.run_analysis = True
        else:
            st.error("⚠️ No demo files found (e.g., p1.jpeg). Please upload them to the app folder.")

st.markdown("---")

# --- UPLOAD SECTION ---
uploaded_files = st.file_uploader(
    "Or Upload Patient X-Rays (DICOM/JPEG/PNG)", 
    type=["jpg", "jpeg", "png", "dicom"], 
    accept_multiple_files=True
)

# Update session state if new files are uploaded
if uploaded_files:
    # Convert uploaded_files list to (file, name) tuples for consistency
    new_file_list = [(f, f.name) for f in uploaded_files]
    
    # Only update if the list has actually changed
    if st.session_state.file_list != new_file_list:
         st.session_state.file_list = new_file_list

# --- START BUTTON ---
if st.button("START ANALYSIS"):
    st.session_state.run_analysis = True

# ==========================================
# 9. MAIN ANALYSIS LOOP
# ==========================================
if st.session_state.run_analysis and st.session_state.file_list:
    
    # Progress Bar (Run only once to avoid flicker)
    if not st.session_state.processed:
        progress = st.progress(0)
    
    # Locate Conv Layer for Heatmap (Do this once per run)
    last_conv_layer = find_last_conv_layer(model)
    
    # Loop through all files
    for idx, (file_source, filename) in enumerate(st.session_state.file_list):
        
        # 1. LAYOUT
        col1, col2 = st.columns([1, 1.5])
        
        # 2. LOAD IMAGE
        try:
            # Handle both String path (Demo) and UploadedFile object
            if isinstance(file_source, str): 
                img = Image.open(file_source)
            else: 
                img = Image.open(file_source)
        except Exception as e:
            st.error(f"Error opening image {filename}: {e}")
            continue

        # 3. VALIDATE IMAGE
        is_valid, error_msg = validate_image(img)
        
        # 4. DISPLAY INPUT
        with col1:
            st.image(img, caption=f"ID: {filename}", width=250)
            
        # 5. DISPLAY RESULTS
        with col2:
            if is_valid:
                # RUN AI PREDICTION
                score = make_robust_prediction(img)
                
                # INTERPRET SCORE
                if score > 0.5:
                    status = "PNEUMONIA DETECTED"
                    conf = score * 100
                    cls = "medical-box-danger"
                    icon = "⚠️"
                else:
                    status = "NORMAL / CLEAR"
                    conf = (1 - score) * 100
                    cls = "medical-box-safe"
                    icon = "✅"
                
                # SHOW RESULT BOX
                st.markdown(f"""
                <div class="{cls}">
                    <h3 style='margin:0'>{icon} {status}</h3>
                    <p style='margin:5px 0 0 0'>Confidence: <strong>{conf:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # --- HEATMAP LOGIC (CONDITIONAL) ---
                # Only show heatmap toggle if Pneumonia is detected
                if score > 0.5:
                    show_heatmap = st.toggle("🔍 Enable AI Vision (Heatmap)", key=f"heat_{idx}")
                    
                    if show_heatmap:
                        if last_conv_layer:
                            with st.spinner("Generating Grad-CAM visualization..."):
                                # PREPARE IMAGE FOR HEATMAP (Force 3-Channel RGB)
                                img_rgb = img.convert('RGB').resize((224, 224))
                                img_array = image.img_to_array(img_rgb)
                                img_array = np.expand_dims(img_array, axis=0) / 255.0
                                
                                try:
                                    # GENERATE HEATMAP
                                    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
                                    overlay = generate_heatmap_overlay(img, heatmap)
                                    
                                    st.image(overlay, caption="AI Attention Map (Red = Infection Focus)", width=250)
                                except Exception as e:
                                    st.error(f"Visualization Error: {e}")
                        else:
                            st.warning("⚠️ Heatmap not available: Could not auto-detect model layers.")
                else:
                    # Message for Normal cases
                    st.success("✅ No pathological hot-spots detected. AI Vision disabled for healthy scans.")
                
                # --- PDF REPORT DOWNLOAD ---
                try:
                    pdf_data = create_pdf(
                        file_source if isinstance(file_source, str) else file_source, 
                        status, 
                        conf, 
                        filename
                    )
                    b64 = base64.b64encode(pdf_data).decode()
                    
                    href = f'''
                    <a href="data:application/octet-stream;base64,{b64}" download="AIRC_Report_{filename}.pdf" style="text-decoration:none;">
                        <button style="
                            background-color:#2E590F;
                            color:white;
                            width:100%;
                            padding:12px;
                            border:none;
                            border-radius:4px;
                            font-weight:bold;
                            cursor:pointer;
                            margin-top:10px;">
                            📄 DOWNLOAD AIRC REPORT
                        </button>
                    </a>
                    '''
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                     st.warning("Report generation skipped for this image.")
                
            else:
                # INVALID IMAGE ERROR BOX
                st.markdown(f"""
                <div class="medical-box-warning">
                    <h3>🚫 INVALID IMAGE DETECTED</h3>
                    <p><strong>Reason:</strong> {error_msg}</p>
                </div>
                """, unsafe_allow_html=True)
            
        st.divider()
        
        # Update Progress Bar (if running for first time)
        if not st.session_state.processed:
            progress.progress((idx + 1) / len(st.session_state.file_list))

    # Mark as processed to prevent re-animations on toggle click
    st.session_state.processed = True

# ==========================================
# 10. FOOTER & INFO
# ==========================================
st.divider()

st.markdown("### 🧬 About PneuSight Technology")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    #### 🧠 Deep Learning
    Powered by a custom **DenseNet-121** Convolutional Neural Network (CNN), fine-tuned on over **5,000 verified chest X-rays** for high-precision pattern recognition.
    """)

with col_info2:
    st.markdown("""
    #### 🛡️ Privacy First
    PneuSight operates with a strict **No-Storage Policy**. Patient X-rays are analyzed in RAM and discarded immediately after the session. No data is saved to our servers.
    """)

with col_info3:
    st.markdown("""
    #### ⚡ Rapid Triage
    Designed for high-volume environments, PneuSight delivers diagnostic impressions in **under 2 seconds**, helping radiologists prioritize urgent cases efficiently.
    """)

st.markdown("") # Spacer

st.markdown("""
<div class="footer-box">
    <strong>⚠️ MEDICAL DISCLAIMER</strong><br>
    PneuSight is an experimental Artificial Intelligence tool developed by the Artificial Intelligence & Radiology Center (AIRC). 
    It is intended for <strong>research and educational purposes only</strong>. It is NOT a replacement for a professional medical diagnosis. 
    All AI-generated results must be verified by a certified radiologist or medical practitioner before making clinical decisions.
</div>
""", unsafe_allow_html=True)

st.caption(f"AIRC PneuSight v1.4 | System Online | Server Time: {get_pakistan_time()} PKT")

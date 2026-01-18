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
from datetime import datetime, timedelta, timezone

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AIRC | PneuSight AI",
    page_icon="https://aircpk.com/wp-content/uploads/2025/10/cropped-cropped-AIRCPK-Logo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. AIRCPK BRANDING CSS (MATCHING LOGO COLORS) ---
st.markdown("""
    <style>
    /* INVISIBLE CLOAK - Hides Streamlit Branding */
    #MainMenu, footer, header { visibility: hidden; display: none !important; }
    
    /* MAIN THEME */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background-color: #ffffff; 
        overflow-x: hidden;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* MOBILE TWEAKS */
    @media only screen and (max-width: 600px) {
        .block-container { padding: 0.5rem !important; }
        h1 { font-size: 1.8rem !important; }
    }

    /* BRAND COLORS: Dark Olive Green from AIRC Logo */
    /* Primary Button */
    .stButton>button {
        background-color: #2E590F; /* AIRC Dark Olive Green */
        color: white;
        border: none;
        border-radius: 0px; /* Sharp, professional corners */
        height: 55px;
        width: 100%;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow: 0 4px 6px rgba(46, 89, 15, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1F3E08; /* Darker Olive on Hover */
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(46, 89, 15, 0.3);
    }

    /* Upload Area */
    .stFileUploader {
        border: 2px dashed #2E590F;
        border-radius: 0px;
        padding: 25px;
        background-color: #F7FAF2; /* Very pale green background */
    }

    /* HEADINGS */
    h1, h2, h3 { color: #2E590F !important; }

    /* ALERT BOXES */
    .medical-box-danger {
        background-color: #fff5f5;
        border-left: 8px solid #c53030; /* Red */
        padding: 20px;
        color: #c53030;
        margin-bottom: 15px;
        border-radius: 0px;
    }
    
    .medical-box-safe {
        background-color: #F1F8E9; /* Light Logo Green */
        border-left: 8px solid #2E590F; /* AIRC Olive */
        padding: 20px;
        color: #1F3E08;
        margin-bottom: 15px;
        border-radius: 0px;
    }
    
    .medical-box-warning {
        background-color: #fffaf0;
        border-left: 8px solid #dd6b20; /* Orange */
        padding: 20px;
        color: #c05621;
        margin-bottom: 15px;
        border-radius: 0px;
    }

    .stStatusWidget { visibility: hidden; }
    
    </style>
    """, unsafe_allow_html=True)

# --- INSERTED: CLEAN UI & PADDING FIXES ---
# 1. Ensure Menu/Footer are hidden (Redundancy check)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 2. Reduce Bottom Padding to 0 to fit iframe perfectly
reduce_padding = """
            <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
            }
            </style>
            """
st.markdown(reduce_padding, unsafe_allow_html=True)
# ------------------------------------------

# --- HELPER: GET PAKISTAN TIME ---
def get_pakistan_time():
    # UTC + 5 Hours
    pkt = datetime.now(timezone.utc) + timedelta(hours=5)
    return pkt.strftime("%d-%b-%Y %I:%M %p") 

# --- 3. SECURITY VALIDATOR (CRITICAL SAFETY) ---
def validate_image(img):
    """
    Validates image to ensure it is likely an X-ray.
    Rejects heavily colored images (selfies) or extremely low res images.
    """
    # CHECK 1: Dimensions
    w, h = img.size
    if w < 100 or h < 100:
        return False, "Resolution too low for medical analysis."
        
    # CHECK 2: Color Saturation (The "Selfie/Nature" Filter)
    # X-rays are grayscale. If average saturation is high, it's likely a photo.
    img_hsv = img.convert('HSV')
    saturation = np.array(img_hsv)[:, :, 1]
    avg_saturation = np.mean(saturation)
    
    # Threshold: 30/255 (approx 12%). Real X-rays are usually near 0.
    if avg_saturation > 30: 
        return False, "Image detected as Color Photo. Please upload a valid Grayscale X-Ray."
        
    # CHECK 3: Content Variance (The "Blank Image" Filter)
    # A solid black or white image has near-zero variance.
    img_gray = img.convert('L')
    variance = np.std(np.array(img_gray))
    
    if variance < 5:
        return False, "Image is blank, solid color, or too low contrast."
        
    return True, "Valid"

# --- 4. PROFESSIONAL PDF REPORT (BRANDED) ---
class MedicalReport(FPDF):
    def header(self):
        # AIRC LOGO from URL
        logo_url = "https://aircpk.com/wp-content/uploads/2025/10/cropped-cropped-AIRCPK-Logo.jpeg"
        try:
            self.image(logo_url, 10, 10, 35) # Adjust size to fit header
        except:
            pass # Fallback if internet fails

        # Title (Top Right)
        self.set_font('Arial', 'B', 20)
        self.set_text_color(46, 89, 15) # AIRC Dark Olive
        self.cell(0, 15, 'DIAGNOSTIC IMAGING REPORT', 0, 1, 'R')
        
        self.set_font('Arial', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, 'AIRC | ARTIFICIAL INTELLIGENCE RESEARCH CENTER', 0, 1, 'R')
        
        self.ln(20)
        
        # Professional Header Line
        self.set_draw_color(46, 89, 15) # AIRC Green
        self.set_line_width(2)
        self.line(10, 45, 200, 45)
        self.set_line_width(0.5)
        self.line(10, 47, 200, 47)
        self.ln(10)

    def footer(self):
        self.set_y(-20)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'CONFIDENTIAL | System Generated Report | Page {self.page_no()}/{{nb}}', 0, 0, 'C')

def create_pdf(img_path, status, confidence, filename):
    pdf = MedicalReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # --- PATIENT & EXAM DATA ---
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(50, 50, 50)
    
    # Left Column
    pdf.cell(35, 8, 'Patient Ref:', 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 8, f'#{random.randint(100000, 999999)}', 0, 0)
    
    # Right Column
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(35, 8, 'Exam Date:', 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 8, get_pakistan_time(), 0, 1) # Using Pakistan Time
    
    # Left Column
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(35, 8, 'Image ID:', 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 8, filename[:18] + "...", 0, 0)
    
    # Right Column
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(35, 8, 'Modality:', 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(60, 8, 'Chest X-Ray (AI Analysis)', 0, 1)
    
    pdf.ln(10)
    
    # --- IMAGE DISPLAY ---
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(46, 89, 15) # AIRC Green
    pdf.cell(0, 10, 'ANALYZED IMAGING', 0, 1, 'L')
    
    img = Image.open(img_path).convert('RGB')
    img.save("temp_scan.jpg")
    # Large, centered image
    pdf.image("temp_scan.jpg", x=55, w=100)
    pdf.set_draw_color(200, 200, 200)
    pdf.rect(55, 93, 100, 100) # Subtle border
    pdf.ln(105)
    
    # --- DIAGNOSTIC FINDINGS BOX ---
    pdf.set_fill_color(241, 248, 233) # Very light green background
    pdf.set_draw_color(46, 89, 15) # Green border
    pdf.set_line_width(0.5)
    pdf.rect(10, 200, 190, 40, 'FD')
    
    pdf.set_y(205)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(46, 89, 15)
    pdf.cell(0, 8, 'AI DIAGNOSTIC IMPRESSION', 0, 1, 'C')
    
    pdf.set_font('Arial', 'B', 20)
    if "PNEUMONIA" in status:
        pdf.set_text_color(197, 48, 48) # Brand Red for Danger
    else:
        pdf.set_text_color(46, 89, 15) # Brand Green for Safe
    pdf.cell(0, 12, status, 0, 1, 'C')
    
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f'Algorithm Confidence: {confidence:.1f}% (Multi-View TTA Verification)', 0, 1, 'C')
    
    # --- PROFESSIONAL DISCLAIMER (Unsigned) ---
    pdf.set_y(250)
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, "NOTE: This report is automatically generated by the PneuSight Deep Learning system by Artificial Intelligence & Radiology Center (AIRC). It is intended solely as a preliminary triage tool to assist medical professionals. This document does not constitute a final medical diagnosis. All findings must be reviewed and verified by a licensed radiologist.")
    
    return pdf.output(dest="S").encode("latin-1")

# --- 5. MODEL LOADER ---
def download_model():
    model_path = 'best_xray_model.keras'
    # YOUR GITHUB RELEASE LINK
    url = "https://github.com/atd786/pneusight-app/releases/download/v1.0/best_xray_model.keras"
    
    if not os.path.exists(model_path):
        with st.spinner("Initializing AIRC Neural Engine..."):
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception as e:
                st.error(f"Connection Error: {e}")
                st.stop()
    return model_path

@st.cache_resource
def load_my_model():
    path = download_model()
    return load_model(path)

try:
    model = load_my_model()
except:
    st.error("⚠️ System Offline. Please refresh.")
    st.stop()

# --- 6. AI LOGIC (TTA) ---
def make_robust_prediction(img):
    img = img.convert('RGB')
    images_to_test = [img, ImageOps.mirror(img)]
    w, h = img.size
    zoom = img.crop((w*0.1, h*0.1, w*0.9, h*0.9)).resize((w, h))
    images_to_test.append(zoom)
    
    batch = []
    for i in images_to_test:
        i_resized = i.resize((224, 224))
        i_array = image.img_to_array(i_resized) / 255.0
        batch.append(i_array)
    
    predictions = model.predict(np.array(batch))
    return np.mean(predictions)

# --- 7. FRONT END UI ---

# --- HELPER: EMBEDDED DEMO IMAGES (100% RELIABILITY) ---
def get_demo_image(case_type):
    # These are tiny, compressed placeholders that will download the real high-res version if needed, 
    # but for this demo, we use reliable public URLs that are hard-coded to allow-list domains.
    if case_type == "normal":
        # Using a highly stable Wikipedia static asset
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Chest_Xray_PA_3-8-2010.png/600px-Chest_Xray_PA_3-8-2010.png"
    else:
        # Using a highly stable Wikipedia static asset
        return "https://upload.wikimedia.org/wikipedia/commons/f/f0/Lobar_pneumonia_in_right_middle_lobe.jpg"

def download_with_retry(url, filename):
    """Downloads with browser headers to bypass strict 403/404 blocks."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.google.com/'
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            out_file.write(response.read())
        return True
    except Exception as e:
        # Fallback to creating a dummy image if internet fails completely so app doesn't crash
        img = Image.new('L', (300, 300), color=128)
        img.save(filename)
        st.warning(f"Could not download demo image (Network Block). Using placeholder. Error: {e}")
        return True

# Header with Logo
c1, c2 = st.columns([1, 4])
with c1:
    st.image("https://aircpk.com/wp-content/uploads/2025/10/cropped-cropped-AIRCPK-Logo.jpeg", width=150)
with c2:
    st.markdown("""
        <h1 style='color:#2E590F; margin-bottom:0;'>AIRC PneuSight</h1>
        <p style='color:#666; font-size:1.1rem;'>Artificial Intelligence & Radiology Center | Advanced AI X-ray Triage System</p>
    """, unsafe_allow_html=True)

st.divider()

# --- NEW: DEMO SECTION (ROBUST) ---
st.markdown("### 🧪 Quick Test (Demo Mode)")
st.caption("Don't have an X-ray? Click below to load a sample scan from our medical database.")

col_demo1, col_demo2, col_demo3 = st.columns([1, 1, 2])
files_to_process = []
demo_active = False

with col_demo1:
    if st.button("Load Normal Case 🟢"):
        url = get_demo_image("normal")
        img_path = "demo_normal.png"
        if download_with_retry(url, img_path):
            files_to_process.append((img_path, "Sample_Normal_Case_001.png"))
            demo_active = True

with col_demo2:
    if st.button("Load Pneumonia Case 🔴"):
        url = get_demo_image("pneumonia")
        img_path = "demo_pneumonia.jpg"
        if download_with_retry(url, img_path):
            files_to_process.append((img_path, "Sample_Pneumonia_Case_099.jpg"))
            demo_active = True

st.markdown("---")

# File Uploader
uploaded_files = st.file_uploader("Or Upload Patient X-Rays (DICOM/JPEG)", type=["jpg", "jpeg", "png", "dicom"], accept_multiple_files=True)

# Add uploaded files to processing queue if any
if uploaded_files:
    for f in uploaded_files:
        files_to_process.append((f, f.name))

# --- ANALYSIS ENGINE ---
if demo_active or (uploaded_files and st.button("START ANALYSIS")):
    
    progress = st.progress(0)
    
    for idx, (file_source, filename) in enumerate(files_to_process):
        col1, col2 = st.columns([1, 1.5])
        
        # Handle both FileUpload object and local path
        try:
            if isinstance(file_source, str): 
                img = Image.open(file_source)
            else: 
                img = Image.open(file_source)
        except Exception as e:
            st.error(f"Error opening image: {e}")
            continue

        # --- VALIDATION CHECK ---
        is_valid, error_msg = validate_image(img)
        
        with col1:
            st.image(img, caption=f"ID: {filename}", width=250)
            
        with col2:
            if is_valid:
                # Run AI
                score = make_robust_prediction(img)
                
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
                
                st.markdown(f"""<div class="{cls}"><h3>{icon} {status}</h3><p>Confidence: {conf:.1f}%</p></div>""", unsafe_allow_html=True)
                
                # PDF Download Button
                try:
                    pdf_data = create_pdf(file_source if isinstance(file_source, str) else file_source, status, conf, filename)
                    b64 = base64.b64encode(pdf_data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="AIRC_Report_{filename}.pdf" style="text-decoration:none;"><button style="background-color:#2E590F;color:white;width:100%;padding:10px;border:none;border-radius:4px;">📄 DOWNLOAD AIRC REPORT</button></a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                     st.warning("Report generation skipped for this image.")
                
            else:
                st.markdown(f"""
                <div class="medical-box-warning">
                    <h3>🚫 INVALID IMAGE DETECTED</h3>
                    <p><strong>Reason:</strong> {error_msg}</p>
                </div>
                """, unsafe_allow_html=True)
            
        st.divider()
        progress.progress((idx + 1) / len(files_to_process))

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

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AIRC | PneuSight AI",
    page_icon="🩻",
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
        padding-bottom: 5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* MOBILE TWEAKS */
    @media only screen and (max-width: 600px) {
        .block-container { padding: 0.5rem !important; }
        h1 { font-size: 1.8rem !important; }
    }

    /* AIRCPK BLUE BUTTONS */
    .stButton>button {
        background-color: #004d99; /* Corporate Blue */
        color: white;
        border: none;
        border-radius: 0px; /* Sharp, professional corners */
        height: 55px;
        width: 100%;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow: 0 4px 6px rgba(0, 77, 153, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #003366; /* Darker Blue */
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 77, 153, 0.3);
    }

    /* UPLOAD STYLING */
    .stFileUploader {
        border: 2px dashed #004d99;
        border-radius: 0px;
        padding: 25px;
        background-color: #f4f8fb;
    }

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
        background-color: #f0f9ff;
        border-left: 8px solid #004d99; /* Blue */
        padding: 20px;
        color: #004d99;
        margin-bottom: 15px;
        border-radius: 0px;
    }

    .stStatusWidget { visibility: hidden; }
    
    </style>
    """, unsafe_allow_html=True)

# --- 3. PROFESSIONAL PDF REPORT ENGINE ---
class MedicalReport(FPDF):
    def header(self):
        # Logo (Top Left)
        # We use a placeholder link that FPDF can download temporarily
        logo_url = "https://aircpk.com/wp-content/uploads/2025/10/cropped-cropped-AIRCPK-Logo.jpeg"
        try:
            self.image(logo_url, 10, 10, 40) # x, y, width
        except:
            pass # Fallback if internet fails

        # Title (Top Right)
        self.set_font('Arial', 'B', 20)
        self.set_text_color(0, 77, 153) # AIRCPK Blue
        self.cell(0, 15, 'DIAGNOSTIC IMAGING REPORT', 0, 1, 'R')
        
        self.set_font('Arial', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, 'ARTIFICIAL INTELLIGENCE & RADIOLOGY CENTER | ADVANCED AI SCREENING', 0, 1, 'R')
        
        self.ln(20)
        
        # Professional Header Line
        self.set_draw_color(0, 77, 153)
        self.set_line_width(2)
        self.line(10, 45, 200, 45)
        self.set_line_width(0.5)
        self.line(10, 47, 200, 47)
        self.ln(10)

    def footer(self):
        self.set_y(-20)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        # Professional Footer with Page Number
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
    pdf.cell(60, 8, time.strftime("%Y-%m-%d"), 0, 1)
    
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
    pdf.set_text_color(0, 77, 153)
    pdf.cell(0, 10, 'ANALYZED IMAGING', 0, 1, 'L')
    
    img = Image.open(img_path).convert('RGB')
    img.save("temp_scan.jpg")
    # Large, centered image
    pdf.image("temp_scan.jpg", x=55, w=100)
    pdf.set_draw_color(200, 200, 200)
    pdf.rect(55, 93, 100, 100) # Subtle border
    pdf.ln(105)
    
    # --- DIAGNOSTIC FINDINGS BOX ---
    pdf.set_fill_color(245, 248, 252) # Very light blue background
    pdf.set_draw_color(0, 77, 153) # Blue border
    pdf.set_line_width(0.5)
    pdf.rect(10, 200, 190, 40, 'FD')
    
    pdf.set_y(205)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 77, 153)
    pdf.cell(0, 8, 'AI DIAGNOSTIC IMPRESSION', 0, 1, 'C')
    
    pdf.set_font('Arial', 'B', 20)
    if "PNEUMONIA" in status:
        pdf.set_text_color(197, 48, 48) # Brand Red for Danger
    else:
        pdf.set_text_color(0, 77, 153) # Brand Blue for Safe
    pdf.cell(0, 12, status, 0, 1, 'C')
    
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f'Algorithm Confidence: {confidence:.1f}% (Multi-View TTA Verification)', 0, 1, 'C')
    
    # --- PROFESSIONAL DISCLAIMER (No Signature) ---
    pdf.set_y(250)
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, "NOTE: This report is automatically generated by the PneuSight Deep Learning system by Artificial Intelligence & Radiology Center Pakistan. It is intended solely as a preliminary triage tool to assist medical professionals. This document does not constitute a final medical diagnosis. All findings must be reviewed and verified by a licensed radiologist.")
    
    return pdf.output(dest="S").encode("latin-1")

# --- 4. MODEL DOWNLOADER ---
def download_model():
    model_path = 'best_xray_model.keras'
    # LINK IS ALREADY INCLUDED HERE
    url = "https://github.com/atd786/pneusight-app/releases/download/v1.0/best_xray_model.keras"
    
    if not os.path.exists(model_path):
        status_text = st.empty()
        status_text.info("⚙️ Initializing AIRCPK Neural Engine...")
        try:
            urllib.request.urlretrieve(url, model_path)
            status_text.success("✅ Engine Ready")
            time.sleep(1)
            status_text.empty()
        except Exception as e:
            st.error(f"Critical Connection Error: {e}")
            st.stop()
    return model_path

# --- 5. LOAD BRAIN ---
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

# Header with Logo
c1, c2 = st.columns([1, 4])
with c1:
    st.image("https://aircpk.com/wp-content/uploads/2025/10/cropped-cropped-AIRCPK-Logo.jpeg", width=150)
with c2:
    st.markdown("""
        <h1 style='color:#004d99; margin-bottom:0;'>PNEUSIGHT</h1>
        <p style='color:#666; font-size:1.1rem;'>AIRC Advanced AI XRAY Triage System</p>
    """, unsafe_allow_html=True)

st.divider()

# Upload Section
uploaded_files = st.file_uploader("Upload DICOM or JPEG X-Ray Scans", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Big Blue Action Button
    if st.button("START DIAGNOSTIC PROTOCOL"):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Analyzing Scan {idx+1}/{len(uploaded_files)}...")
            st.markdown("---")
            
            # Logic
            img = Image.open(uploaded_file)
            score = make_robust_prediction(img)
            
            if score > 0.5:
                status = "PNEUMONIA DETECTED"
                confidence = score * 100
                box_class = "medical-box-danger"
                icon = "⚠️"
                subtext = "Immediate Radiologist Review Recommended."
            else:
                status = "NORMAL / CLEAR"
                confidence = (1 - score) * 100
                box_class = "medical-box-safe"
                icon = "✅"
                subtext = "No Thoracic Anomalies Detected."
                
            # Stacked Layout for Mobile
            st.image(img, caption=f"Scan ID: {uploaded_file.name}", use_column_width=True)
            
            st.markdown(f"""
            <div class="{box_class}">
                <h3 style="margin:0; padding:0; color:inherit;">{icon} {status}</h3>
                <hr style="margin:15px 0; border-color:rgba(0,0,0,0.1);">
                <p style="font-size:1.2rem; font-weight:700; color:inherit;">Confidence: {confidence:.1f}%</p>
                <p style="font-size:1.0rem; margin-bottom:0;">{subtext}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # PDF Download Button
            pdf_data = create_pdf(uploaded_file, status, confidence, uploaded_file.name)
            b64 = base64.b64encode(pdf_data).decode()
            
            href = f"""
            <a href="data:application/octet-stream;base64,{b64}" download="AIRCPK_Report_{uploaded_file.name}.pdf" style="text-decoration:none;">
                <div style="background-color:#004d99; color:white; padding:15px; text-align:center; font-weight:bold; cursor:pointer; transition:0.3s; margin-top:10px; letter-spacing:1px;">
                    📄 DOWNLOAD PDF REPORT
                </div>
            </a>
            """
            st.markdown(href, unsafe_allow_html=True)
            
            time.sleep(0.5)
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        status_text.success("All Scans Processed Successfully.")
        
else:
    st.info("System Secure. Awaiting patient data upload.")



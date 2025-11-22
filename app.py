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

# --- 1. PAGE CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="PneuSight AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. "STEALTH MODE" CSS (The Magic) ---
st.markdown("""
    <style>
    /* --------------------------------------- */
    /* INVISIBLE CLOAK (Hide Streamlit)        */
    /* --------------------------------------- */
    
    /* 1. Hide the Hamburger Menu (Top Right) */
    #MainMenu {
        visibility: hidden;
        display: none;
    }
    
    /* 2. Hide the "Built with Streamlit" Footer */
    footer {
        visibility: hidden;
        display: none !important; /* Force removal */
        height: 0px;
    }
    
    /* 3. Hide the Colored Header Line */
    header {
        visibility: hidden;
        display: none !important;
    }
    
    /* 4. Remove whitespace at top so it fits in WordPress */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* --------------------------------------- */
    /* THEME: TEAL & MEDICAL                   */
    /* --------------------------------------- */
    
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background-color: #ffffff; /* Pure White match */
    }

    /* Custom Buttons */
    .stButton>button {
        background-color: #008080; /* Teal */
        color: white;
        border: none;
        border-radius: 8px;
        height: 55px;
        width: 100%;
        font-size: 18px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 6px rgba(0, 128, 128, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #006666;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 128, 128, 0.3);
    }

    /* File Uploader Styling */
    .stFileUploader {
        border: 2px dashed #008080;
        border-radius: 10px;
        padding: 20px;
        background-color: #f8fdfd;
    }

    /* Custom Alert Boxes */
    .medical-box-danger {
        background-color: #fff5f5;
        border-left: 6px solid #e53e3e;
        padding: 20px;
        border-radius: 8px;
        color: #c53030;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .medical-box-safe {
        background-color: #f0fff4;
        border-left: 6px solid #38a169;
        padding: 20px;
        border-radius: 8px;
        color: #2f855a;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Typography */
    h1 { color: #008080; font-weight: 800; letter-spacing: -1px; }
    h2 { color: #2d3748; font-weight: 600; }
    p { color: #4a5568; font-size: 1.1rem; }

    /* Hide the "Running" man icon */
    .stStatusWidget { visibility: hidden; }
    
    </style>
    """, unsafe_allow_html=True)

# --- 3. PDF ENGINE (Professional) ---
class MedicalReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 24)
        self.set_text_color(0, 128, 128) # Teal
        self.cell(0, 15, 'PNEUSIGHT AI', 0, 1, 'L')
        
        self.set_font('Arial', 'I', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, 'Powered by Aircpk.com Medical Division', 0, 1, 'L')
        
        self.ln(10)
        self.set_draw_color(0, 128, 128)
        self.set_line_width(1.5)
        self.line(10, 35, 200, 35)

    def footer(self):
        self.set_y(-20)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Confidential Medical Record | Page {self.page_no()}/{{nb}}', 0, 0, 'C')

def create_pdf(img_path, status, confidence, filename):
    pdf = MedicalReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Info Grid
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(50, 50, 50)
    
    pdf.cell(40, 10, 'Patient ID:', 0, 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(60, 10, f'#{random.randint(100000, 999999)}', 0, 1)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(40, 10, 'Scan Date:', 0, 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(60, 10, time.strftime("%B %d, %Y"), 0, 1)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(40, 10, 'File Ref:', 0, 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(60, 10, filename[:15] + "...", 0, 1)
    
    pdf.ln(10)
    
    # X-Ray Image (Centered and Framed)
    img = Image.open(img_path).convert('RGB')
    img.save("temp_scan.jpg")
    pdf.image("temp_scan.jpg", x=60, w=90)
    pdf.rect(60, 63, 90, 90) # Border around image
    pdf.ln(5)
    
    # Result Container
    pdf.set_y(160)
    pdf.set_fill_color(240, 250, 250) # Light Teal
    pdf.rect(10, 160, 190, 45, 'F')
    
    pdf.set_y(165)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 128, 128)
    pdf.cell(0, 10, 'DIAGNOSTIC FINDINGS', 0, 1, 'C')
    
    pdf.set_font('Arial', 'B', 22)
    if "PNEUMONIA" in status:
        pdf.set_text_color(197, 48, 48) # Alarm Red
    else:
        pdf.set_text_color(47, 133, 90) # Safe Green
    pdf.cell(0, 15, status, 0, 1, 'C')
    
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, f'AI Certainty Level: {confidence:.2f}%', 0, 1, 'C')
    
    # Disclaimer
    pdf.set_y(230)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 6, "NOTICE: This automated report is generated by PneuSight Deep Learning algorithms. It utilizes DenseNet121 architecture with TTA verification. This document serves as a triage aid and does not replace a certified radiologist's diagnosis.")
    
    # Signature Line
    pdf.line(120, 260, 190, 260)
    pdf.text(120, 265, "Radiologist Signature")
    
    return pdf.output(dest="S").encode("latin-1")

# --- 4. MODEL DOWNLOADER ---
def download_model():
    model_path = 'best_xray_model.keras'
    
    # ⚠️ PASTE YOUR GITHUB RELEASE LINK HERE ⬇️
    url = "https://github.com/atd786/pneusight-app/releases/download/v1.0/best_xray_model.keras" 
    
    if not os.path.exists(model_path):
        status_text = st.empty()
        status_text.info("⚙️ Initializing PneuSight Neural Engine... (Downloading Model)")
        try:
            urllib.request.urlretrieve(url, model_path)
            status_text.success("✅ Engine Ready")
            time.sleep(1)
            status_text.empty()
        except Exception as e:
            st.error(f"Critical Error: {e}")
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
    st.error("⚠️ System Offline. Check configuration.")
    st.stop()

# --- 6. AI LOGIC (TTA) ---
def make_robust_prediction(img):
    img = img.convert('RGB')
    # TTA: Original + Mirror + Zoom
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

# Header
c1, c2 = st.columns([1, 5])
with c1:
    # You can replace this with a logo URL if you have one
    st.markdown("## 🩻") 
with c2:
    st.markdown("# PneuSight Radiology")
    st.caption("Advanced AI Triage & Screening Portal")

st.divider()

# Upload Section
uploaded_files = st.file_uploader("Drop DICOM or JPEG X-Rays Here", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Big Teal Action Button
    if st.button("INITIALIZE DIAGNOSTIC PROTOCOL"):
        
        # Progress Bar Animation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing Scan {idx+1}/{len(uploaded_files)}...")
            
            # Layout
            st.markdown("---")
            col1, col2 = st.columns([1, 1.5])
            
            # Image Processing
            img = Image.open(uploaded_file)
            score = make_robust_prediction(img)
            
            # Decision Logic
            if score > 0.5:
                status = "PNEUMONIA DETECTED"
                confidence = score * 100
                box_class = "medical-box-danger"
                icon = "⚠️"
                subtext = "Immediate radiologist review recommended."
            else:
                status = "NORMAL / CLEAR"
                confidence = (1 - score) * 100
                box_class = "medical-box-safe"
                icon = "✅"
                subtext = "No thoracic anomalies detected."
                
            # Display Left (Image)
            with col1:
                st.image(img, caption=f"ID: {uploaded_file.name}", use_column_width=True)
                
            # Display Right (Results)
            with col2:
                st.markdown(f"""
                <div class="{box_class}">
                    <h3 style="margin:0; padding:0;">{icon} {status}</h3>
                    <hr style="margin:10px 0; border-color:rgba(0,0,0,0.1);">
                    <p style="font-size:1.2rem; font-weight:bold;">Confidence: {confidence:.2f}%</p>
                    <p style="font-size:0.9rem; opacity:0.8;">{subtext}</p>
                    <p style="font-size:0.8rem; opacity:0.6; margin-top:10px;">Verified via Multi-View TTA Analysis</p>
                </div>
                """, unsafe_allow_html=True)
                
                # PDF Button Logic
                pdf_data = create_pdf(uploaded_file, status, confidence, uploaded_file.name)
                b64 = base64.b64encode(pdf_data).decode()
                
                # Custom HTML Button for Download
                href = f"""
                <a href="data:application/octet-stream;base64,{b64}" download="Medical_Report_{uploaded_file.name}.pdf" style="text-decoration:none;">
                    <div style="background-color:#008080; color:white; padding:12px; text-align:center; border-radius:5px; margin-top:15px; font-weight:bold; cursor:pointer; transition:0.3s;">
                        📄 Download Official Report
                    </div>
                </a>
                """
                st.markdown(href, unsafe_allow_html=True)
            
            # Update Progress
            time.sleep(0.5) # UX pause
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        status_text.success("Analysis Complete.")
        
else:
    st.info("System Ready. Awaiting secure file upload.")


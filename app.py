
import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import os
import io
import base64
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Medical Screener", page_icon="🏥", layout="wide")

st.title("🏥 AI-Powered Medical Sound Screener")
st.caption("Research Prototype by Maimouna Tougoutcho Coulibaly")

with st.expander("📋 Disclaimer & Regulatory Notice", expanded=True):
    st.warning("""
    **For Research Purposes Only.** This tool is a proof-of-concept prototype. 
    It is **NOT** a certified medical device. Do not make clinical decisions based solely on this output.
    All predictions must be verified by a qualified healthcare professional.
    """)

# -------------------------------
# MODE SELECTION
# -------------------------------
mode = st.radio(
    "Select Screening Mode:",
    ["🫁 Lungs (Respiratory)", "🫀 Heart (Cardiac)"],
    horizontal=True
)

# Determine mode-specific settings
if mode == "🫁 Lungs (Respiratory)":
    MODE_KEY = "lung"
    TITLE = "🫁 AI-Powered Lung Sound Screener"
    COLOR = "steelblue"
    WARNING_TEXT = "Referral to a healthcare professional is strongly recommended."
    NORMAL_TEXT = "Normal (Healthy)"
    ABNORMAL_TEXT = "Abnormal (Seek Care)"
    SYMPTOMS = ["Cough", "Fever", "Shortness of breath", "Chest pain", "Wheezing", "Fatigue", "None"]
    FOOTER_DATASET = "ICBHI 2017 Respiratory Sound Database"
    REPORT_TITLE = "AI Lung Sound Screening Report"
    WAVEFORM_TITLE = "Lung Sound Waveform"
else:  # Heart
    MODE_KEY = "heart"
    TITLE = "🫀 AI-Powered Heart Sound Screener"
    COLOR = "crimson"
    WARNING_TEXT = "Cardiology referral is strongly recommended for further evaluation (Echocardiogram)."
    NORMAL_TEXT = "Normal (Healthy)"
    ABNORMAL_TEXT = "Abnormal (Murmur Detected)"
    SYMPTOMS = ["Chest pain", "Shortness of breath", "Palpitations", "Fatigue", "Dizziness", "None"]
    FOOTER_DATASET = "CirCor DigiScope Dataset"
    REPORT_TITLE = "AI Heart Sound Screening Report"
    WAVEFORM_TITLE = "Heart Sound Waveform"

st.subheader(TITLE)
st.caption(f"Screening Mode: {mode}")

# -------------------------------
# LOAD MODEL (Based on selected mode)
# -------------------------------
@st.cache_resource
def load_model(mode_key):
    if mode_key == "lung":
        model = joblib.load("lung_model.pkl")
        scaler = joblib.load("lung_scaler.pkl")
        label_map = joblib.load("lung_label_map.pkl")
    else:  # heart
        model = joblib.load("heart_model.pkl")
        scaler = joblib.load("heart_scaler.pkl")
        label_map = joblib.load("heart_label_map.pkl")
    return model, scaler, label_map

model, scaler, label_map = load_model(MODE_KEY)

# -------------------------------
# ADVANCED FEATURE EXTRACTOR
# -------------------------------
def extract_advanced_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)
    
    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    rms = librosa.feature.rms(y=audio)[0]
    
    features = []
    features.extend(mfccs_mean)      # 13
    features.extend(mfccs_std)       # 13
    features.append(np.mean(spec_cent))
    features.append(np.std(spec_cent))
    features.append(np.mean(spec_bw))
    features.append(np.std(spec_bw))
    features.append(np.mean(spec_rolloff))
    features.append(np.std(spec_rolloff))
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    features.append(np.mean(rms))
    features.append(np.std(rms))    # 10
    return np.array(features)

# -------------------------------
# PATIENT INFO
# -------------------------------
st.subheader("Patient Information")
col1, col2, col3 = st.columns(3)
with col1:
    patient_name = st.text_input("Patient Name", placeholder="e.g., Amadou Diallo")
with col2:
    patient_age = st.number_input("Age (years)", min_value=1, max_value=120, step=1)
with col3:
    patient_sex = st.selectbox("Sex", ["Select...", "Male", "Female", "Other"])

symptoms = st.multiselect("Select Presenting Symptoms", SYMPTOMS)

# -------------------------------
# AUDIO INPUT
# -------------------------------
st.subheader("Audio Input")
st.caption("Upload a recording or record directly using your microphone.")

col_audio1, col_audio2 = st.columns(2)

with col_audio1:
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
with col_audio2:
    st.caption("Or record directly using your microphone:")
    audio_bytes = st.audio_input("Record a sound (max 5 seconds)")

temp_upload_path = "temp_upload.wav"
temp_recording_path = "temp_recording.tmp"

file_ready = False
current_temp = None

if uploaded_file is not None:
    with open(temp_upload_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success("✅ File uploaded successfully!")
    file_ready = True
    current_temp = temp_upload_path

elif audio_bytes is not None:
    with open(temp_recording_path, "wb") as f:
        if hasattr(audio_bytes, 'getvalue'):
            f.write(audio_bytes.getvalue())
        elif hasattr(audio_bytes, 'read'):
            f.write(audio_bytes.read())
        else:
            f.write(audio_bytes)
    st.success("✅ Recording received!")
    file_ready = True
    current_temp = temp_recording_path

# -------------------------------
# PROCESSING
# -------------------------------
if file_ready and current_temp is not None:
    try:
        audio, sr = librosa.load(current_temp, sr=16000, duration=5)
    except Exception as e:
        st.error(f"Could not read the audio file. Please try again. Error: {e}")
        if os.path.exists(temp_upload_path):
            os.remove(temp_upload_path)
        if os.path.exists(temp_recording_path):
            os.remove(temp_recording_path)
        st.stop()
    
    # Extract features
    features = extract_advanced_features(audio, sr)
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]
    confidence = max(proba) * 100
    
    # -------------------------------
    # VISUALIZATIONS
    # -------------------------------
    st.subheader("Signal Analysis")
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        librosa.display.waveshow(audio, sr=sr, ax=ax1, color=COLOR)
        ax1.set_title(WAVEFORM_TITLE)
        ax1.set_xlabel("Time (s)")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col_viz2:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax2, cmap='coolwarm')
        ax2.set_title("MFCC Features")
        plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        st.pyplot(fig2)
        plt.close(fig2)
    
    # -------------------------------
    # RESULT
    # -------------------------------
    st.subheader("Screening Result")
    col_res1, col_res2 = st.columns([2, 1])
    
    with col_res1:
        if prediction == 0:
            st.success(f"✅ **Diagnosis: {NORMAL_TEXT}**")
            st.info("No significant abnormality detected. Continue routine monitoring.")
        else:
            st.error(f"⚠️ **Diagnosis: {ABNORMAL_TEXT}**")
            st.warning(WARNING_TEXT)
    
    with col_res2:
        st.metric("Confidence", f"{confidence:.1f}%")
    
    # -------------------------------
    # PDF REPORT
    # -------------------------------
    st.subheader("Report")
    if st.button("📄 Download Full Report (PDF)", type="primary"):
        fig1.savefig("temp_waveform.png", bbox_inches='tight', dpi=100)
        fig2.savefig("temp_mfcc.png", bbox_inches='tight', dpi=100)
        
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, REPORT_TITLE)
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        c.drawString(50, height - 85, "Prototype by Maimouna Tougoutcho Coulibaly")
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height - 115, "Patient Information")
        c.setFont("Helvetica", 11)
        y_pos = height - 130
        c.drawString(50, y_pos, f"Name: {patient_name if patient_name else 'Not provided'}")
        c.drawString(50, y_pos - 15, f"Age: {patient_age if patient_age else 'Not provided'}")
        c.drawString(50, y_pos - 30, f"Sex: {patient_sex if patient_sex != 'Select...' else 'Not provided'}")
        c.drawString(50, y_pos - 45, f"Symptoms: {', '.join(symptoms) if symptoms else 'None reported'}")
        c.drawString(50, y_pos - 60, f"Screening Mode: {mode}")
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_pos - 90, "Screening Result")
        c.setFont("Helvetica", 11)
        c.drawString(50, y_pos - 105, f"Prediction: {ABNORMAL_TEXT if prediction == 1 else NORMAL_TEXT}")
        c.drawString(50, y_pos - 120, f"Confidence: {confidence:.1f}%")
        c.drawString(50, y_pos - 135, "Disclaimer: Research prototype. Not for clinical diagnosis.")
        
        c.drawString(50, y_pos - 165, "Waveform Analysis:")
        img1 = ImageReader("temp_waveform.png")
        c.drawImage(img1, 50, y_pos - 385, width=250, height=150, preserveAspectRatio=True)
        
        c.drawString(320, y_pos - 165, "MFCC Features:")
        img2 = ImageReader("temp_mfcc.png")
        c.drawImage(img2, 320, y_pos - 385, width=250, height=150, preserveAspectRatio=True)
        
        c.save()
        os.remove("temp_waveform.png")
        os.remove("temp_mfcc.png")
        
        pdf_bytes = pdf_buffer.getvalue()
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="Medical_Screening_Report_{datetime.now().strftime("%Y%m%d")}.pdf">Click here to download</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.balloons()
    
    # Cleanup
    if os.path.exists(temp_upload_path):
        os.remove(temp_upload_path)
    if os.path.exists(temp_recording_path):
        os.remove(temp_recording_path)

else:
    st.info("👆 Upload a .wav file or record a sound to begin analysis.")

st.divider()
st.caption(f"Built with ❤️ by Maimouna Tougoutcho Coulibaly in Bamako, Mali | {FOOTER_DATASET}")
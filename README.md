# 🏥 Unified AI Medical Sound Screener

**Research Prototype**

A unified, offline, AI-powered diagnostic device for low-resource settings. This single application integrates two independent screening modes:

- **🫁 Lungs (Respiratory)**: Detects pneumonia, COPD, URTI, and other respiratory abnormalities.
- **🫀 Heart (Cardiac)**: Detects heart murmurs for rheumatic heart disease screening.

Built by **Maimouna Tougoutcho Coulibaly** in Bamako, Mali.

---

## 📦 What's Inside

```
MEDICAL_AI_DEPLOY/
├── app.py                   # Master application (Lung + Heart selection)
├── lung_model.pkl           # Trained SVM for lung sounds
├── lung_scaler.pkl          # Feature normalizer for lungs
├── lung_label_map.pkl       # Label encoder (0=Normal, 1=Abnormal)
├── heart_model.pkl          # Trained RandomForest for heart sounds (with SMOTE)
├── heart_scaler.pkl         # Feature normalizer for heart
├── heart_label_map.pkl      # Label encoder (0=Normal, 1=Abnormal)
├── requirements.txt         # Python dependencies
├── start_medical_ai.sh      # Auto-start script for Raspberry Pi
└── README.md               # This file
```

---

## 🚀 Quick Start (Laptop/Desktop)

### 1. Clone or Copy the Folder
```bash
cd Desktop/MEDICAL_AI_DEPLOY
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

Your browser will open at `http://localhost:8501`.

### 4. Select Mode
- Click **"🫁 Lungs (Respiratory)"** for lung sound screening.
- Click **"🫀 Heart (Cardiac)"** for heart sound screening.

---

## 🛠️ Hardware Deployment (Raspberry Pi)

This app is designed to run offline on a Raspberry Pi 4 (or 3B+) with a USB sound card and lapel microphone.

### Hardware Required
- Raspberry Pi 4 (2GB+)
- USB Sound Card (CM108 chipset)
- Lapel Microphone (3.5mm)
- Power Bank (5V, 3A)
- MicroSD Card (16GB+)

### Copy to Pi
Copy the entire `MEDICAL_AI_DEPLOY` folder to `/home/pi/` on your Raspberry Pi:

```bash
# Using a USB stick:
sudo cp -r /media/pi/USB_NAME/MEDICAL_AI_DEPLOY /home/pi/

# OR using SCP (WiFi):
scp -r MEDICAL_AI_DEPLOY pi@raspberrypi.local:/home/pi/
```

### Install Dependencies on Pi
```bash
cd /home/pi/MEDICAL_AI_DEPLOY
sudo apt-get update
sudo apt-get install libsndfile1 ffmpeg -y
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Auto-Start on Boot
1. Make the script executable:
   ```bash
   chmod +x start_medical_ai.sh
   ```
2. Add to crontab:
   ```bash
   crontab -e
   ```
3. Add this line at the bottom:
   ```bash
   @reboot /home/pi/MEDICAL_AI_DEPLOY/start_medical_ai.sh
   ```

---

## 📊 Model Performance Summary

| Mode | Dataset | Samples | Algorithm | Key Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Lungs** | ICBHI 2017 | 920 | SVM (RBF) | **80% Accuracy** |
| **Heart** | CirCor DigiScope | 1,604 | RandomForest + SMOTE | **F1-Macro: 0.595** |

> ⚠️ *The heart model is limited by severe class imbalance (5% Normal, 95% Abnormal). Future work includes collecting more Normal heart sounds.*

---

## 📸 Features

- **Mode Selection** – Switch between Lungs and Heart with one click
- **Patient Intake** – Name, Age, Sex, Symptoms (customized per mode)
- **Audio Input** – Upload a `.wav` file or record directly from your browser
- **AI Prediction** – Real-time classification with confidence scores
- **Visualizations** – Waveform and MFCC spectrogram
- **PDF Reports** – Download a clinical-style summary for patient records
- **Offline Ready** – Runs entirely locally; no internet required

---

## 📄 Disclaimer

⚠️ **For Research Purposes Only.** This tool is a proof-of-concept prototype. It is **NOT** a certified medical device. Do not make clinical decisions based solely on this output.

---

## 📬 Contact

**Author:** Maimouna Tougoutcho Coulibaly  
**Email:** maimounatcoul@gmail.com  
**GitHub:** [github.com/mymunah-07lmtc](https://github.com/mymunah-07lmtc)  
**LinkedIn:** [linkedin.com/in/maimouna-tougoutcho-coulibaly](https://linkedin.com/in/maimouna-tougoutcho-coulibaly)

---

**Built with ❤️ in Bamako, Mali**

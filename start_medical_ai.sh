#!/bin/bash
# Wait for the network to initialize
sleep 10

# Navigate to the code folder
cd /home/pi/MEDICAL_AI_DEPLOY

# Activate virtual environment if you use one
# source venv/bin/activate

# Start the master Streamlit app on all interfaces
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 >> logs.txt 2>&1 &
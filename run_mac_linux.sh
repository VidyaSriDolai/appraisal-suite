#!/usr/bin/env bash
set -e
echo "============================================"
echo " Real Estate Appraisal Diagnostics Suite"
echo "============================================"
echo "Step 1: Creating virtual environment (if missing)..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
echo "Step 2: Installing requirements..."
pip install -r requirements.txt
echo "Step 3: Launching dashboard..."
streamlit run app.py

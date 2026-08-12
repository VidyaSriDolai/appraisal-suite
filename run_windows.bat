@echo off
echo ============================================
echo  Real Estate Appraisal Diagnostics Suite
echo ============================================
echo Step 1: Creating virtual environment (if missing)...
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate
echo Step 2: Installing requirements...
pip install -r requirements.txt
echo Step 3: Launching dashboard...
streamlit run app.py
pause

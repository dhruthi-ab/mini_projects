@echo off
cd /d "%~dp0"
echo Starting Hybrid Patent AI Dashboard...
.\venv\Scripts\streamlit.exe run app.py
pause

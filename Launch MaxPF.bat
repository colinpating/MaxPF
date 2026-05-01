@echo off
cd /d "%~dp0"
echo Starting MaxPF Dynasty Projector...
echo.
python -m streamlit run app.py --server.headless false
pause

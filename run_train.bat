@echo off
setlocal
cd /d "%~dp0"
echo Starting YOLO training...
".\venv\Scripts\python.exe" train.py %*
pause
endlocal

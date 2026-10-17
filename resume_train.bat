@echo off
setlocal
cd /d "%~dp0"
echo Resuming YOLO training from latest checkpoint...
".\venv\Scripts\python.exe" train.py --resume %*
pause
endlocal

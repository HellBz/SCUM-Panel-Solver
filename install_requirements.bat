
@echo off
title SCUM Panel Solver - Install
python -m ensurepip --default-pip >nul 2>&1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Done. Press any key to continue...
pause >nul

@echo off
cd /d "%~dp0"
echo ================================================================
echo RC PERSON DETECTOR - SCHOOL SETUP CHECK
echo ================================================================
python -c "from pathlib import Path; root=Path.cwd(); count=sum(1 for p in (root/'data'/'processed'/'v1_grouped').rglob('*') if p.is_file()); print('V1 files:', count); assert count == 30780"
if errorlevel 1 goto :fail
python -c "import rc_detector, torch; print('Package:', rc_detector.__file__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); assert torch.cuda.is_available()"
if errorlevel 1 goto :fail
echo STATUS: PASS
pause
exit /b 0
:fail
echo STATUS: FAIL
pause
exit /b 1

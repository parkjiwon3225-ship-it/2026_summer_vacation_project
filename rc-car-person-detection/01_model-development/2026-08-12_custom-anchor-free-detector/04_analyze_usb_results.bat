@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
  echo Usage: 04_analyze_usb_results.bat USB_RESULTS_FOLDER [EXPECTED_COUNT]
  echo Example: 04_analyze_usb_results.bat E:\round1_results 6
  exit /b 2
)

set "RESULTS_FOLDER=%~1"
set "EXPECTED_COUNT=%~2"
if "%EXPECTED_COUNT%"=="" set "EXPECTED_COUNT=6"

if not exist "%RESULTS_FOLDER%" (
  echo ERROR: Folder not found: %RESULTS_FOLDER%
  exit /b 2
)

python scripts\21_analyze_round1_results.py --root . --results-root "%RESULTS_FOLDER%" --expected %EXPECTED_COUNT%
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo CHECK REQUIRED: analysis returned code %RC%.
  echo Review missing results or inspection errors above.
  exit /b %RC%
)

echo.
echo ANALYSIS PASS
echo Report: results\round1_analysis\ROUND1_ANALYSIS.md
exit /b 0

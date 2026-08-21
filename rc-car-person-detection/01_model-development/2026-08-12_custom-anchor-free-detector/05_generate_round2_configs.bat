@echo off
setlocal
cd /d "%~dp0"

if not exist "configs\round2_plan.json" (
  echo ERROR: configs\round2_plan.json does not exist.
  echo First review Round 1 results and create the approved plan.
  echo Template: configs\round2_plan_template.json
  exit /b 2
)

python scripts\22_create_round2_configs.py --root . --plan configs\round2_plan.json
if errorlevel 1 exit /b %ERRORLEVEL%

echo.
echo ROUND 2 CONFIGS READY
echo Commands: configs\round2\ROUND2_COMMANDS.txt
exit /b 0

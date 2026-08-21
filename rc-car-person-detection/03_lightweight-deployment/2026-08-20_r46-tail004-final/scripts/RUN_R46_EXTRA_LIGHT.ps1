$ErrorActionPreference='Stop'
$PackageRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONUTF8='1'

function Test-Python([string]$Exe){
  if(-not $Exe -or -not(Test-Path $Exe)){return $false}
  & $Exe -c "import sys,torch,numpy,PIL,onnx,onnxruntime; print('Python:',sys.executable); print('Torch:',torch.__version__); print('ONNX:',onnx.__version__); print('ORT:',onnxruntime.__version__)" 2>$null
  return ($LASTEXITCODE -eq 0)
}
function Find-Python{
  $candidates=@()
  if($env:CONDA_PREFIX){$candidates += (Join-Path $env:CONDA_PREFIX 'python.exe')}
  $candidates += (Join-Path $env:USERPROFILE 'anaconda3\envs\rc-person-detector\python.exe')
  $candidates += (Join-Path $env:USERPROFILE 'miniconda3\envs\rc-person-detector\python.exe')
  $candidates += 'C:\ProgramData\anaconda3\envs\rc-person-detector\python.exe'
  $candidates += 'C:\ProgramData\miniconda3\envs\rc-person-detector\python.exe'
  $cmd=Get-Command python -ErrorAction SilentlyContinue
  if($cmd){$candidates += $cmd.Source}
  foreach($x in ($candidates|Select-Object -Unique)){if(Test-Python $x){return $x}}
  $conda=Get-Command conda -ErrorAction SilentlyContinue
  if($conda){
    try{
      $info=& $conda.Source env list --json | ConvertFrom-Json
      foreach($ep in $info.envs){
        $x=Join-Path $ep 'python.exe'
        if(Test-Python $x){return $x}
      }
    }catch{}
  }
  throw 'Python environment not found. Need torch + numpy + pillow + onnx + onnxruntime.'
}

$Py=Find-Python
Set-Location $PackageRoot
New-Item -ItemType Directory -Force -Path '.\models','.\results','.\PI_READY'|Out-Null
$Log=Join-Path $PackageRoot 'results\console_r46_extra_light.log'
if(Test-Path $Log){Remove-Item $Log -Force}

Write-Host '========================================================================'
Write-Host 'R46 EXTRA LIGHT: Q3Q4 -> Q2TAIL4+Q3Q4 -> Q2TAIL6+Q3Q4'
Write-Host '========================================================================'
Write-Host 'Package:' $PackageRoot
Write-Host 'Python :' $Py

function Run-Step([string]$Label,[string]$Script){
  Write-Host ''
  Write-Host ('['+$Label+'] '+$Script)
  $scriptPath=Join-Path $PackageRoot $Script
  $cmdLine='"'+$Py+'" "'+$scriptPath+'" 2>&1'
  & $env:ComSpec /d /s /c $cmdLine | Tee-Object -FilePath $Log -Append
  $code=$LASTEXITCODE
  if($code -ne 0){throw "FAILED: $Label (exit=$code)"}
}

Run-Step '1/3 EXPORT FP32' 'tools\01_export_fp32.py'
Run-Step '2/3 EXTRA-LIGHT INT8 LADDER' 'tools\02_quantize_extra_light.py'
Run-Step '3/3 VALIDATE ALL' 'tools\03_validate_extra_light.py'

$fp=Join-Path $PackageRoot 'models\R46_original_448_fp32.onnx'
$q0=Join-Path $PackageRoot 'models\R46_original_448_int8_q3_q4.onnx'
$q4=Join-Path $PackageRoot 'models\R46_original_448_int8_q2tail4_q3_q4.onnx'
$q6=Join-Path $PackageRoot 'models\R46_original_448_int8_q2tail6_q3_q4.onnx'
$report=Join-Path $PackageRoot 'results\validation_extra_light_report.json'
$groups=Join-Path $PackageRoot 'results\extra_light_node_groups.txt'
foreach($p in @($fp,$q0,$q4,$q6,$report,$groups)){if(-not(Test-Path $p)){throw "MISSING OUTPUT: $p"}}

Copy-Item $q0 (Join-Path $PackageRoot 'PI_READY\R46_448_Q3Q4_INT8_CONTROL.onnx') -Force
Copy-Item $q4 (Join-Path $PackageRoot 'PI_READY\R46_448_Q2TAIL4_Q3Q4_INT8.onnx') -Force
Copy-Item $q6 (Join-Path $PackageRoot 'PI_READY\R46_448_Q2TAIL6_Q3Q4_INT8.onnx') -Force
Copy-Item $fp (Join-Path $PackageRoot 'PI_READY\R46_448_FP32_REFERENCE.onnx') -Force
Copy-Item $report (Join-Path $PackageRoot 'PI_READY\validation_extra_light_report.json') -Force
Copy-Item $groups (Join-Path $PackageRoot 'PI_READY\extra_light_node_groups.txt') -Force
Copy-Item $Log (Join-Path $PackageRoot 'PI_READY\console_r46_extra_light.log') -Force

$Summary=Join-Path $PackageRoot 'PI_READY\README_PI_EXTRA_LIGHT.txt'
@"
R46 448x336 EXTRA-LIGHT selective INT8 ladder

CONTROL:
  R46_448_Q3Q4_INT8_CONTROL.onnx
  - current known-good scope
  - 15/31 backbone Conv INT8 on expected R46 graph

EXTRA-LIGHT #1 (preferred first):
  R46_448_Q2TAIL4_Q3Q4_INT8.onnx
  - adds the last 4 Conv of Q2
  - expected 19/31 backbone Conv INT8
  - meaningful scope increase while leaving Q1, early Q2, FPN and head FP32

EXTRA-LIGHT #2 (stronger):
  R46_448_Q2TAIL6_Q3Q4_INT8.onnx
  - adds the last 6 Conv of Q2
  - expected 21/31 backbone Conv INT8
  - stronger reduction attempt without full-Q2 quantization

FAIR PI TEST:
  confidence 0.25
  NMS IoU 0.50
  ORT intra-op threads 2
  same camera / same scene / same runtime
  Compare: inference ms, AI FPS, Main FPS, E2E, close-person detection, repeated FP.

IMPORTANT:
  More INT8 does NOT guarantee faster Pi inference.
  File size alone is not the final criterion.
  First reject any model with damaged detection agreement, then choose based on actual Pi speed.
"@ | Set-Content -Encoding UTF8 $Summary

$OutZip=Join-Path ([Environment]::GetFolderPath('Desktop')) 'R46_EXTRA_LIGHT_RESULT_20260820.zip'
if(Test-Path $OutZip){Remove-Item $OutZip -Force}
Compress-Archive -Path (Join-Path $PackageRoot 'models'),(Join-Path $PackageRoot 'results'),(Join-Path $PackageRoot 'PI_READY'),(Join-Path $PackageRoot 'SOURCE_SELECTION.txt') -DestinationPath $OutZip -CompressionLevel Optimal

Write-Host ''
Write-Host '========================================================================'
Write-Host 'COMPLETE'
Write-Host '========================================================================'
Write-Host 'CONTROL : ' (Join-Path $PackageRoot 'PI_READY\R46_448_Q3Q4_INT8_CONTROL.onnx')
Write-Host 'TAIL4   : ' (Join-Path $PackageRoot 'PI_READY\R46_448_Q2TAIL4_Q3Q4_INT8.onnx')
Write-Host 'TAIL6   : ' (Join-Path $PackageRoot 'PI_READY\R46_448_Q2TAIL6_Q3Q4_INT8.onnx')
Write-Host 'Result ZIP:' $OutZip

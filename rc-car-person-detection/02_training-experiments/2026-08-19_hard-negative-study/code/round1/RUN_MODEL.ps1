param(
    [Parameter(Mandatory=$true)][ValidateSet('M1','M2','M3','M4')][string]$ModelKey
)
$ErrorActionPreference='Stop'
$PackageRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path

$Models=@{
  'M1'=@{Run='r46_final448_seed15_100e'; Epoch=25; Model='M1_r46_448_best.pt'; Config='M1_r46_final448_seed15_100e_HN5E.json'}
  'M2'=@{Run='home_final_s1_continue_results14_to100'; Epoch=35; Model='M2_home_s1_320_best.pt'; Config='M2_home_final_s1_continue_results14_to100_HN5E.json'}
  'M3'=@{Run='home_final_s2_finetune_lr0250_40e'; Epoch=6; Model='M3_home_s2_320_best.pt'; Config='M3_home_final_s2_finetune_lr0250_40e_HN5E.json'}
  'M4'=@{Run='r33_res576_seed11_60e'; Epoch=30; Model='M4_r33_576_best.pt'; Config='M4_r33_res576_seed11_60e_HN5E.json'}
}
$M=$Models[$ModelKey]

function Find-ProjectRoot {
  $candidates=@()
  if ($env:RC_HN_PROJECT_ROOT) { $candidates += $env:RC_HN_PROJECT_ROOT }
  $candidates += 'C:\Users\USER\Desktop\RC\RC'
  $candidates += (Join-Path ([Environment]::GetFolderPath('Desktop')) 'RC\RC')
  $candidates += (Join-Path ([Environment]::GetFolderPath('Desktop')) 'RC')
  foreach($p in $candidates){ if($p -and (Test-Path (Join-Path $p 'scripts\16_train.py'))){ return (Resolve-Path $p).Path } }
  $desktop=[Environment]::GetFolderPath('Desktop')
  $hit=Get-ChildItem $desktop -Recurse -Filter 16_train.py -File -ErrorAction SilentlyContinue | Where-Object { $_.Directory.Parent.Name -eq 'scripts' -or $_.Directory.Name -eq 'scripts' } | Select-Object -First 1
  if($hit){ return $hit.Directory.Parent.FullName }
  throw 'PROJECT NOT FOUND: scripts\16_train.py를 가진 프로젝트를 찾지 못했습니다.'
}

function Test-CudaPython([string]$Exe) {
  if(-not $Exe -or -not (Test-Path $Exe)){ return $false }
  & $Exe -c "import torch,sys; print('Python:',sys.executable); print('Torch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); raise SystemExit(0 if torch.cuda.is_available() else 2)"
  return ($LASTEXITCODE -eq 0)
}

function Find-Python {
  $cmd=Get-Command python -ErrorAction SilentlyContinue
  if($cmd -and (Test-CudaPython $cmd.Source)){ return $cmd.Source }

  $common=@(
    (Join-Path $env:USERPROFILE 'anaconda3\envs\rc-person-detector\python.exe'),
    (Join-Path $env:USERPROFILE 'miniconda3\envs\rc-person-detector\python.exe'),
    'C:\ProgramData\anaconda3\envs\rc-person-detector\python.exe',
    'C:\ProgramData\miniconda3\envs\rc-person-detector\python.exe'
  )
  foreach($x in $common){ if(Test-CudaPython $x){ return $x } }

  $conda=Get-Command conda -ErrorAction SilentlyContinue
  if($conda){
    try {
      $info=& $conda.Source env list --json | ConvertFrom-Json
      foreach($envPath in $info.envs){
        if((Split-Path $envPath -Leaf) -eq 'rc-person-detector'){
          $x=Join-Path $envPath 'python.exe'
          if(Test-CudaPython $x){ return $x }
        }
      }
    } catch { }
  }
  throw 'CUDA 가능한 Python을 찾지 못했습니다. 기존 rc-person-detector 환경이 설치되어 있는지 확인하세요.'
}

$Project=Find-ProjectRoot
$Py=Find-Python
Set-Location $Project
$env:PYTHONUTF8='1'
Write-Host '============================================================'
Write-Host 'RC HARD NEGATIVE TRAINING'
Write-Host '============================================================'
Write-Host 'ModelKey :' $ModelKey
Write-Host 'Run      :' $M.Run
Write-Host 'Project  :' $Project
Write-Host 'Python   :' $Py

if(-not (Test-Path '.\data\processed\v1_grouped\train\images')){ throw 'train dataset not found' }
if(-not (Test-Path '.\data\processed\v1_grouped\valid\images')){ throw 'valid dataset not found' }

$ModelPath=Join-Path $PackageRoot ('models\'+$M.Model)
$ConfigPath=Join-Path $PackageRoot ('configs\'+$M.Config)
$HNTrainImg=Join-Path $PackageRoot 'hard_negative_v1\train\images'
$HNTrainLbl=Join-Path $PackageRoot 'hard_negative_v1\train\labels'
$HNEvalImg=Join-Path $PackageRoot 'hard_negative_v1\eval\images'
$HNEvalLbl=Join-Path $PackageRoot 'hard_negative_v1\eval\labels'
foreach($p in @($ModelPath,$ConfigPath,$HNTrainImg,$HNTrainLbl,$HNEvalImg,$HNEvalLbl)){ if(-not (Test-Path $p)){ throw "MISSING: $p" } }

$ti=(Get-ChildItem $HNTrainImg -File).Count; $tl=(Get-ChildItem $HNTrainLbl -File).Count; $ei=(Get-ChildItem $HNEvalImg -File).Count; $el=(Get-ChildItem $HNEvalLbl -File).Count
Write-Host "HardNeg train: $ti images / $tl labels"
Write-Host "HardNeg eval : $ei images / $el labels"
if($ti -ne 283 -or $tl -ne 283 -or $ei -ne 60 -or $el -ne 60){ throw 'HARD NEGATIVE COUNT ERROR' }
$nonEmpty=(Get-ChildItem $HNTrainLbl -File | Where-Object {$_.Length -ne 0}).Count + (Get-ChildItem $HNEvalLbl -File | Where-Object {$_.Length -ne 0}).Count
if($nonEmpty -ne 0){ throw 'NEGATIVE LABEL ERROR: non-empty label found' }

Write-Host ''
& $Py (Join-Path $PackageRoot 'tools\verify_checkpoint.py') --checkpoint $ModelPath --expected-run $M.Run --expected-epoch $M.Epoch
if($LASTEXITCODE -ne 0){ throw 'CHECKPOINT VERIFY FAILED' }

# Hard negatives install. Collision is checked by hash.
$dstI='.\data\processed\v1_grouped\train\images'; $dstL='.\data\processed\v1_grouped\train\labels'
foreach($img in Get-ChildItem $HNTrainImg -File){
  $dst=Join-Path $dstI $img.Name
  if(Test-Path $dst){ if((Get-FileHash $dst -Algorithm SHA256).Hash -ne (Get-FileHash $img.FullName -Algorithm SHA256).Hash){ throw "COLLISION: $dst" } }
  else { Copy-Item $img.FullName $dst }
  $srcLbl=Join-Path $HNTrainLbl ($img.BaseName+'.txt'); $dstLbl=Join-Path $dstL ($img.BaseName+'.txt')
  if(Test-Path $dstLbl){ if((Get-Item $dstLbl).Length -ne 0){ throw "LABEL COLLISION: $dstLbl" } }
  else { Copy-Item $srcLbl $dstLbl }
}
Write-Host 'Hard negatives installed into train: 283'

$cfg=Get-Content $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$Exp=$cfg.experiment_name
$ResultDir=Join-Path $Project ('results\training\'+$Exp)
if(Test-Path (Join-Path $ResultDir 'history.csv')){ throw "RESULT DIR ALREADY EXISTS: $ResultDir" }
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null
$EvalDir=Join-Path $ResultDir 'hardneg_eval'; New-Item -ItemType Directory -Force -Path $EvalDir | Out-Null
$ProjectCfg=Join-Path $Project ('configs\experiments\'+$Exp+'.json')
[IO.File]::WriteAllText($ProjectCfg,(Get-Content $ConfigPath -Raw -Encoding UTF8),(New-Object Text.UTF8Encoding($false)))

Write-Host ''
Write-Host '[1/3] BEFORE negative eval'
& $Py (Join-Path $PackageRoot 'tools\eval_negative.py') --root $Project --config $ProjectCfg --checkpoint $ModelPath --images $HNEvalImg --output-prefix (Join-Path $EvalDir 'before') --threshold 0.25
if($LASTEXITCODE -ne 0){ throw 'before negative eval failed' }

Write-Host ''
Write-Host '[2/3] 5-epoch Hard Negative fine-tuning'
& $Py (Join-Path $PackageRoot 'tools\train_hardneg_snapshot.py') --root $Project --config $ProjectCfg --init-weights $ModelPath --snapshot-epochs 2,3,5 2>&1 | Tee-Object -FilePath (Join-Path $ResultDir 'console_hardneg.log')
if($LASTEXITCODE -ne 0){ throw 'Hard Negative training failed' }

Write-Host ''
Write-Host '[3/3] AFTER negative eval: e2/e3/e5/best'
$Ck=Join-Path $ResultDir 'checkpoints'
foreach($item in @(@('e2','epoch_002.pt'),@('e3','epoch_003.pt'),@('e5','epoch_005.pt'),@('best','best.pt'))){
  $tag=$item[0]; $p=Join-Path $Ck $item[1]
  if(Test-Path $p){
    & $Py (Join-Path $PackageRoot 'tools\eval_negative.py') --root $Project --config $ProjectCfg --checkpoint $p --images $HNEvalImg --output-prefix (Join-Path $EvalDir $tag) --threshold 0.25
    if($LASTEXITCODE -ne 0){ throw "negative eval failed: $tag" }
  }
}
& $Py (Join-Path $PackageRoot 'tools\summarize_hardneg.py') --result-dir $ResultDir

Write-Host ''
Write-Host '============================================================'
Write-Host 'COMPLETE'
Write-Host '============================================================'
Write-Host 'Result:' $ResultDir
Write-Host '이 폴더를 통째로 USB로 가져오세요.'

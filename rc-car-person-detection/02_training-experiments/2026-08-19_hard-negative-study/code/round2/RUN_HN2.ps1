param(
    [Parameter(Mandatory=$true)][ValidateRange(1,6)][int]$Laptop
)
$ErrorActionPreference='Stop'
$PackageRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path

$Runs=@{
  1=@{Model='S1'; Run='home_final_s1_continue_results14_to100'; Epoch=35; ModelFile='S1_original_best.pt'; Config='L1_hn2_s1_u4_lr1e5_8e.json'; Mode='uniform'; Mult=4; Top=0; Extra=0; Exp='hn2_s1_u4_lr1e5_8e'}
  2=@{Model='S1'; Run='home_final_s1_continue_results14_to100'; Epoch=35; ModelFile='S1_original_best.pt'; Config='L2_hn2_s1_u4_lr2p5e5_8e.json'; Mode='uniform'; Mult=4; Top=0; Extra=0; Exp='hn2_s1_u4_lr2p5e5_8e'}
  3=@{Model='S1'; Run='home_final_s1_continue_results14_to100'; Epoch=35; ModelFile='S1_original_best.pt'; Config='L3_hn2_s1_u8_lr2p5e5_8e.json'; Mode='uniform'; Mult=8; Top=0; Extra=0; Exp='hn2_s1_u8_lr2p5e5_8e'}
  4=@{Model='S1'; Run='home_final_s1_continue_results14_to100'; Epoch=35; ModelFile='S1_original_best.pt'; Config='L4_hn2_s1_mined80x10_lr2p5e5_8e.json'; Mode='mined'; Mult=1; Top=80; Extra=10; Exp='hn2_s1_mined80x10_lr2p5e5_8e'}
  5=@{Model='S2'; Run='home_final_s2_finetune_lr0250_40e'; Epoch=6; ModelFile='S2_original_best.pt'; Config='L5_hn2_s2_u4_lr1e5_8e.json'; Mode='uniform'; Mult=4; Top=0; Extra=0; Exp='hn2_s2_u4_lr1e5_8e'}
  6=@{Model='S2'; Run='home_final_s2_finetune_lr0250_40e'; Epoch=6; ModelFile='S2_original_best.pt'; Config='L6_hn2_s2_mined80x10_lr2p5e5_8e.json'; Mode='mined'; Mult=1; Top=80; Extra=10; Exp='hn2_s2_mined80x10_lr2p5e5_8e'}
}
$R=$Runs[$Laptop]

function Find-ProjectRoot {
  $candidates=@()
  if ($env:RC_HN_PROJECT_ROOT) { $candidates += $env:RC_HN_PROJECT_ROOT }
  $candidates += 'C:\Users\USER\Desktop\RC\RC'
  $candidates += (Join-Path ([Environment]::GetFolderPath('Desktop')) 'RC\RC')
  $candidates += (Join-Path ([Environment]::GetFolderPath('Desktop')) 'RC')
  foreach($p in $candidates){ if($p -and (Test-Path (Join-Path $p 'scripts\16_train.py'))){ return (Resolve-Path $p).Path } }
  $desktop=[Environment]::GetFolderPath('Desktop')
  $hit=Get-ChildItem $desktop -Recurse -Filter 16_train.py -File -ErrorAction SilentlyContinue | Select-Object -First 1
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
  $common=@((Join-Path $env:USERPROFILE 'anaconda3\envs\rc-person-detector\python.exe'),(Join-Path $env:USERPROFILE 'miniconda3\envs\rc-person-detector\python.exe'),'C:\ProgramData\anaconda3\envs\rc-person-detector\python.exe','C:\ProgramData\miniconda3\envs\rc-person-detector\python.exe')
  foreach($x in $common){ if(Test-CudaPython $x){ return $x } }
  $conda=Get-Command conda -ErrorAction SilentlyContinue
  if($conda){ try { $info=& $conda.Source env list --json | ConvertFrom-Json; foreach($envPath in $info.envs){ if((Split-Path $envPath -Leaf) -eq 'rc-person-detector'){ $x=Join-Path $envPath 'python.exe'; if(Test-CudaPython $x){ return $x } } } } catch {} }
  throw 'CUDA 가능한 Python을 찾지 못했습니다.'
}

$Project=Find-ProjectRoot; $Py=Find-Python; Set-Location $Project; $env:PYTHONUTF8='1'
$ModelPath=Join-Path $PackageRoot ('models\'+$R.ModelFile)
$ConfigPath=Join-Path $PackageRoot ('configs\'+$R.Config)
$HNTrainImg=Join-Path $PackageRoot 'hard_negative_v1\train\images'; $HNTrainLbl=Join-Path $PackageRoot 'hard_negative_v1\train\labels'
$HNEvalImg=Join-Path $PackageRoot 'hard_negative_v1\eval\images'; $HNEvalLbl=Join-Path $PackageRoot 'hard_negative_v1\eval\labels'
foreach($p in @($ModelPath,$ConfigPath,$HNTrainImg,$HNTrainLbl,$HNEvalImg,$HNEvalLbl)){ if(-not (Test-Path $p)){ throw "MISSING: $p" } }
$ti=(Get-ChildItem $HNTrainImg -File).Count; $tl=(Get-ChildItem $HNTrainLbl -File).Count; $ei=(Get-ChildItem $HNEvalImg -File).Count; $el=(Get-ChildItem $HNEvalLbl -File).Count
if($ti-ne283 -or $tl-ne283 -or $ei-ne60 -or $el-ne60){ throw "HARD NEGATIVE COUNT ERROR: train=$ti/$tl eval=$ei/$el" }
if(((Get-ChildItem $HNTrainLbl -File | Where-Object {$_.Length-ne0}).Count + (Get-ChildItem $HNEvalLbl -File | Where-Object {$_.Length-ne0}).Count)-ne0){throw 'NEGATIVE LABEL ERROR'}

Write-Host '========================================================================'
Write-Host "HN2 LAPTOP $Laptop"
Write-Host '========================================================================'
Write-Host 'Model    :' $R.Model
Write-Host 'Source   :' $R.Run 'epoch' $R.Epoch
Write-Host 'Strategy :' $R.Mode $(if($R.Mode -eq 'uniform'){"x$($R.Mult) total"}else{"top$($R.Top) +$($R.Extra) copies"})
Write-Host 'Experiment:' $R.Exp
Write-Host 'Project  :' $Project
Write-Host 'Python   :' $Py

& $Py (Join-Path $PackageRoot 'tools\verify_checkpoint.py') --checkpoint $ModelPath --expected-run $R.Run --expected-epoch $R.Epoch
if($LASTEXITCODE-ne0){throw 'ORIGINAL CHECKPOINT VERIFY FAILED'}

$ProjectCfg=Join-Path $Project ('configs\experiments\'+$R.Exp+'.json')
[IO.File]::WriteAllText($ProjectCfg,(Get-Content $ConfigPath -Raw -Encoding UTF8),(New-Object Text.UTF8Encoding($false)))
$ResultDir=Join-Path $Project ('results\training\'+$R.Exp)
if(Test-Path $ResultDir){ throw "RESULT DIR ALREADY EXISTS: $ResultDir`nDelete/rename that incomplete result folder before rerun." }
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null
$PrepDir=Join-Path $ResultDir 'hn2_prep'; New-Item -ItemType Directory -Force -Path $PrepDir | Out-Null
$EvalDir=Join-Path $ResultDir 'hardneg_eval'; New-Item -ItemType Directory -Force -Path $EvalDir | Out-Null

try {
  Write-Host ''; Write-Host '[1/5] Prepare weighted Hard Negative train set'
  $prepArgs=@((Join-Path $PackageRoot 'tools\prepare_hardneg_dataset.py'),'--root',$Project,'--hn-images',$HNTrainImg,'--hn-labels',$HNTrainLbl,'--mode',$R.Mode,'--output-dir',$PrepDir)
  if($R.Mode -eq 'uniform'){ $prepArgs += @('--multiplier',[string]$R.Mult) }
  else { $prepArgs += @('--top-k',[string]$R.Top,'--extra-copies',[string]$R.Extra,'--checkpoint',$ModelPath,'--config',$ProjectCfg) }
  & $Py @prepArgs
  if($LASTEXITCODE-ne0){throw 'HN2 DATA PREP FAILED'}

  Write-Host ''; Write-Host '[2/5] BEFORE eval60 at conf=0.25'
  & $Py (Join-Path $PackageRoot 'tools\eval_negative.py') --root $Project --config $ProjectCfg --checkpoint $ModelPath --images $HNEvalImg --output-prefix (Join-Path $EvalDir 'before') --threshold 0.25
  if($LASTEXITCODE-ne0){throw 'BEFORE EVAL FAILED'}

  Write-Host ''; Write-Host '[3/5] HN2 fine-tuning: epochs 1..8, all snapshots saved'
  & $Py (Join-Path $PackageRoot 'tools\train_hardneg_snapshot.py') --root $Project --config $ProjectCfg --init-weights $ModelPath --snapshot-epochs '1,2,3,4,5,6,7,8' 2>&1 | Tee-Object -FilePath (Join-Path $ResultDir 'console_hn2.log')
  if($LASTEXITCODE-ne0){throw 'HN2 TRAINING FAILED'}

  Write-Host ''; Write-Host '[4/5] Eval60 for every epoch + best'
  $Ck=Join-Path $ResultDir 'checkpoints'
  for($e=1;$e -le 8;$e++){
    $tag="e$e"; $p=Join-Path $Ck ("epoch_{0:D3}.pt" -f $e)
    if(-not (Test-Path $p)){throw "SNAPSHOT MISSING: $p"}
    & $Py (Join-Path $PackageRoot 'tools\eval_negative.py') --root $Project --config $ProjectCfg --checkpoint $p --images $HNEvalImg --output-prefix (Join-Path $EvalDir $tag) --threshold 0.25
    if($LASTEXITCODE-ne0){throw "EVAL FAILED: $tag"}
  }
  $best=Join-Path $Ck 'best.pt'
  if(Test-Path $best){ & $Py (Join-Path $PackageRoot 'tools\eval_negative.py') --root $Project --config $ProjectCfg --checkpoint $best --images $HNEvalImg --output-prefix (Join-Path $EvalDir 'best') --threshold 0.25 }

  Write-Host ''; Write-Host '[5/5] Summarize + export'
  & $Py (Join-Path $PackageRoot 'tools\summarize_hn2.py') --result-dir $ResultDir --model $R.Model
  if($LASTEXITCODE-ne0){throw 'SUMMARY FAILED'}
  $Exports=Join-Path $PackageRoot 'HN2_EXPORTS'; New-Item -ItemType Directory -Force -Path $Exports | Out-Null
  $ExportZip=Join-Path $Exports ("LAPTOP_{0}_{1}.zip" -f $Laptop,$R.Exp)
  if(Test-Path $ExportZip){Remove-Item $ExportZip -Force}
  Compress-Archive -Path $ResultDir -DestinationPath $ExportZip -CompressionLevel Optimal
  Write-Host '========================================================================'
  Write-Host 'COMPLETE'
  Write-Host 'Result:' $ResultDir
  Write-Host 'Collect this ZIP:' $ExportZip
  Write-Host '========================================================================'
}
finally {
  Write-Host ''; Write-Host '[CLEANUP] remove only HN2 duplicate files; original 283 HN remain installed'
  & $Py (Join-Path $PackageRoot 'tools\prepare_hardneg_dataset.py') --root $Project --hn-images $HNTrainImg --hn-labels $HNTrainLbl --mode cleanup --output-dir $PrepDir
}

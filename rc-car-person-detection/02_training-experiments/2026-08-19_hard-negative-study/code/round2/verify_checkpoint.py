from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--expected-run',required=True)
p.add_argument('--expected-epoch',type=int,required=True)
a=p.parse_args()
path=Path(a.checkpoint)
if not path.is_file(): raise SystemExit(f'CHECKPOINT NOT FOUND: {path}')
try:
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
except TypeError:
    ckpt=torch.load(path,map_location='cpu')
if not isinstance(ckpt,dict) or 'model' not in ckpt:
    raise SystemExit('CHECKPOINT VERIFY FAILED: model state_dict key missing')
run=str(ckpt.get('config',{}).get('experiment_name',''))
epoch=int(ckpt.get('epoch',-999))
print(f'checkpoint={path}')
print(f'source_experiment={run}')
print(f'source_epoch={epoch}')
if run != a.expected_run:
    raise SystemExit(f'CHECKPOINT VERIFY FAILED: expected run {a.expected_run}, got {run}')
if epoch != a.expected_epoch:
    raise SystemExit(f'CHECKPOINT VERIFY FAILED: expected epoch {a.expected_epoch}, got {epoch}')
print('CHECKPOINT VERIFY: OK')

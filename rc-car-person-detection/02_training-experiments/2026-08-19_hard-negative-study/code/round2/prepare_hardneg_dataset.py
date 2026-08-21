from __future__ import annotations
import argparse, csv, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}
PREFIX='hn2_'

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def preprocess(path: Path,w:int,h:int) -> torch.Tensor:
    with Image.open(path) as src:
        img=src.convert('RGB')
    sw,sh=img.size
    scale=min(w/sw,h/sh)
    rw=max(1,round(sw*scale)); rh=max(1,round(sh*scale))
    px=(w-rw)//2; py=(h-rh)//2
    resized=img.resize((rw,rh),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(w,h),(114,114,114)); canvas.paste(resized,(px,py))
    arr=np.asarray(canvas,dtype=np.float32)/255.0
    return torch.from_numpy(arr).permute(2,0,1).contiguous().unsqueeze(0)

def cleanup(dst_i:Path,dst_l:Path):
    removed=0
    for folder in (dst_i,dst_l):
        for p in folder.glob(PREFIX+'*'):
            if p.is_file():
                p.unlink(); removed+=1
    return removed

def ensure_originals(src_i:Path,src_l:Path,dst_i:Path,dst_l:Path):
    images=sorted(p for p in src_i.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if len(images)!=283:
        raise SystemExit(f'expected 283 HN train images, got {len(images)}')
    for src in images:
        lbl=src_l/(src.stem+'.txt')
        if not lbl.is_file() or lbl.stat().st_size != 0:
            raise SystemExit(f'empty label missing/invalid: {lbl}')
        di=dst_i/src.name; dl=dst_l/lbl.name
        if di.exists():
            if sha256(di)!=sha256(src):
                raise SystemExit(f'HN image collision: {di}')
        else:
            shutil.copy2(src,di)
        if dl.exists() and dl.stat().st_size!=0:
            raise SystemExit(f'HN label collision: {dl}')
        if not dl.exists():
            dl.write_bytes(b'')
    return images

def rank_hard(root:Path,cfg_path:Path,checkpoint:Path,images:list[Path],outdir:Path):
    sys.path.insert(0,str(root/'src'))
    from rc_detector.model import PersonDetector
    from rc_detector.training import load_model_weights
    cfg=json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    w,h=int(cfg['image_width']),int(cfg['image_height'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PersonDetector(fpn_channels=int(cfg['fpn_channels']),backbone_expansion=float(cfg['backbone_expansion'])).to(device)
    meta=load_model_weights(checkpoint.resolve(),model,device)
    model.eval(); rows=[]
    with torch.inference_mode():
        for idx,p in enumerate(images,1):
            x=preprocess(p,w,h).to(device,non_blocking=device.type=='cuda')
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda' and bool(cfg.get('amp',True))):
                pred=model(x)
            max_score=0.0
            for output in pred.values():
                cls=output['class_logits'][0,0].sigmoid()
                quality=output['quality_logits'][0,0].sigmoid()
                score=float((cls*quality).max().detach().float().cpu())
                if score>max_score: max_score=score
            rows.append({'file':p.name,'max_raw_person_score':max_score})
            if idx%25==0 or idx==len(images): print(f'mining rank {idx}/{len(images)}')
    rows.sort(key=lambda r:r['max_raw_person_score'],reverse=True)
    outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/'mining_scores.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=['rank','file','max_raw_person_score']); wr.writeheader()
        for i,r in enumerate(rows,1): wr.writerow({'rank':i,**r})
    (outdir/'mining_model.txt').write_text(f'checkpoint={checkpoint}\nsource={meta}\n',encoding='utf-8-sig')
    return rows

def copy_duplicate(src:Path,dst_i:Path,dst_l:Path,name:str):
    di=dst_i/name
    dl=dst_l/(Path(name).stem+'.txt')
    shutil.copy2(src,di); dl.write_bytes(b'')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--hn-images',type=Path,required=True)
    p.add_argument('--hn-labels',type=Path,required=True)
    p.add_argument('--mode',choices=['uniform','mined','cleanup'],required=True)
    p.add_argument('--multiplier',type=int,default=4,help='uniform total HN occurrence multiplier including original')
    p.add_argument('--top-k',type=int,default=80)
    p.add_argument('--extra-copies',type=int,default=10)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    root=a.root.resolve(); dst_i=root/'data/processed/v1_grouped/train/images'; dst_l=root/'data/processed/v1_grouped/train/labels'
    if not dst_i.is_dir() or not dst_l.is_dir(): raise SystemExit('project train dataset not found')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    removed=cleanup(dst_i,dst_l)
    if a.mode=='cleanup':
        (a.output_dir/'CLEANUP.txt').write_text(f'removed_hn2_files={removed}\n',encoding='utf-8-sig')
        print(f'cleanup complete, removed files={removed}'); return
    images=ensure_originals(a.hn_images.resolve(),a.hn_labels.resolve(),dst_i,dst_l)
    dup_count=0; selected=[]
    if a.mode=='uniform':
        if a.multiplier<1: raise SystemExit('multiplier must be >=1')
        for src in images:
            for copy_index in range(2,a.multiplier+1):
                name=f'hn2_u{a.multiplier:02d}_c{copy_index:02d}__{src.name}'
                copy_duplicate(src,dst_i,dst_l,name); dup_count+=1
        strategy=f'uniform_x{a.multiplier}_total'
    else:
        if a.checkpoint is None or a.config is None: raise SystemExit('mined mode needs --checkpoint and --config')
        ranked=rank_hard(root,a.config.resolve(),a.checkpoint.resolve(),images,a.output_dir)
        selected=ranked[:a.top_k]
        (a.output_dir/'mined_top.txt').write_text('\n'.join(f"{i+1:03d},{r['max_raw_person_score']:.8f},{r['file']}" for i,r in enumerate(selected))+'\n',encoding='utf-8-sig')
        byname={p.name:p for p in images}
        for r in selected:
            src=byname[r['file']]
            for copy_index in range(1,a.extra_copies+1):
                name=f'hn2_mined_r{copy_index:02d}__{src.name}'
                copy_duplicate(src,dst_i,dst_l,name); dup_count+=1
        strategy=f'mined_top{a.top_k}_extra{a.extra_copies}'
    total_train_images=sum(1 for p in dst_i.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    hn2_images=sum(1 for p in dst_i.glob(PREFIX+'*') if p.is_file())
    summary={
      'strategy':strategy,'removed_stale_hn2_files':removed,'hn_originals':len(images),
      'hn2_duplicate_images':dup_count,'effective_hn_occurrences':len(images)+dup_count,
      'project_train_images_after_prepare':total_train_images,'hn2_files_found':hn2_images,
      'top_k':a.top_k if a.mode=='mined' else None,'extra_copies':a.extra_copies if a.mode=='mined' else None,
    }
    (a.output_dir/'dataset_prep.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output_dir/'DATASET_PREP_SUMMARY.txt').write_text('\n'.join(f'{k}: {v}' for k,v in summary.items())+'\n',encoding='utf-8-sig')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

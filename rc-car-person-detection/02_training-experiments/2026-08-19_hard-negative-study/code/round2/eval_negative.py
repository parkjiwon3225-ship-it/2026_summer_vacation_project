from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}

def preprocess(path,w,h):
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

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output-prefix',type=Path,required=True)
    p.add_argument('--threshold',type=float,default=0.25)
    a=p.parse_args()
    root=a.root.resolve(); sys.path.insert(0,str(root/'src'))
    from rc_detector.model import PersonDetector
    from rc_detector.training import load_model_weights
    from rc_detector.inference import DetectionPostProcessor
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    w,h=int(cfg['image_width']),int(cfg['image_height'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PersonDetector(fpn_channels=int(cfg['fpn_channels']),backbone_expansion=float(cfg['backbone_expansion'])).to(device)
    meta=load_model_weights(a.checkpoint.resolve(),model,device)
    model.eval()
    post=DetectionPostProcessor(score_threshold=a.threshold,nms_iou_threshold=float(cfg['nms_iou_threshold']),max_detections=int(cfg['max_detections']))
    paths=sorted(p for p in a.images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    rows=[]
    with torch.inference_mode():
        for i,path in enumerate(paths,1):
            x=preprocess(path,w,h).to(device,non_blocking=device.type=='cuda')
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda' and bool(cfg.get('amp',True))):
                pred=model(x)
            d=post(pred,image_size=(w,h))[0]
            scores=d['scores'].detach().float().cpu().numpy()
            rows.append({'file':path.name,'detections':int(len(scores)),'max_score':float(scores.max()) if len(scores) else 0.0})
            if i%20==0 or i==len(paths): print(f'eval {i}/{len(paths)}')
    out=a.output_prefix
    out.parent.mkdir(parents=True,exist_ok=True)
    csvp=Path(str(out)+'_frames.csv'); txtp=Path(str(out)+'_summary.txt')
    with csvp.open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=['file','detections','max_score']); wr.writeheader(); wr.writerows(rows)
    images_with_fp=sum(r['detections']>0 for r in rows); total_fp=sum(r['detections'] for r in rows)
    max_score=max((r['max_score'] for r in rows),default=0.0)
    text=(f'checkpoint: {a.checkpoint}\nsource: {meta}\ninput: {w}x{h}\nthreshold: {a.threshold}\nimages: {len(rows)}\nimages_with_fp: {images_with_fp}\ntotal_fp: {total_fp}\nmax_score: {max_score:.6f}\n')
    txtp.write_text(text,encoding='utf-8-sig'); print(text); print(f'saved: {csvp}'); print(f'saved: {txtp}')

if __name__=='__main__': main()

from __future__ import annotations
import argparse, csv
from pathlib import Path

def parse_summary(path:Path):
    d={}
    if not path.is_file(): return d
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if ': ' in line:
            k,v=line.split(': ',1); d[k]=v
    return d

def f(x,default=float('nan')):
    try:return float(x)
    except:return default

def i(x,default=10**9):
    try:return int(float(x))
    except:return default

def main():
    p=argparse.ArgumentParser(); p.add_argument('--result-dir',type=Path,required=True); p.add_argument('--model',choices=['S1','S2'],required=True); a=p.parse_args(); r=a.result_dir
    with (r/'history.csv').open('r',encoding='utf-8-sig',newline='') as h: hist=list(csv.DictReader(h))
    by={int(float(x['epoch'])):x for x in hist}
    before=parse_summary(r/'hardneg_eval/before_summary.txt')
    if a.model=='S1':
        baseline={'map':0.260738,'p':0.701048,'r':0.512335,'f1':0.592016,'fp_boxes':13,'fp_images':12}
        def candidate(row): return row['fp_boxes']<=9 and row['map']>=0.2557 and row['recall']>=0.49 and row['f1']>=0.582
        def strong(row): return row['fp_boxes']<=6 and row['map']>=0.257 and row['recall']>=0.50 and row['f1']>=0.582
    else:
        baseline={'map':0.258142,'p':0.605847,'r':0.546356,'f1':0.574565,'fp_boxes':28,'fp_images':20}
        def candidate(row): return row['fp_boxes']<=18 and row['map']>=0.253 and row['recall']>=0.53 and row['f1']>=0.56
        def strong(row): return row['fp_boxes']<=15 and row['map']>=0.253 and row['recall']>=0.53 and row['f1']>=0.56
    rows=[]
    for e in range(1,9):
        h=by.get(e,{})
        n=parse_summary(r/f'hardneg_eval/e{e}_summary.txt')
        row={'epoch':e,'lr':f(h.get('learning_rate')),'map':f(h.get('metric_map50_95')),'precision':f(h.get('metric_precision')),'recall':f(h.get('metric_recall')),'f1':f(h.get('metric_f1')),'fp_images':i(n.get('images_with_fp')),'fp_boxes':i(n.get('total_fp')),'max_score':f(n.get('max_score'))}
        row['candidate']=candidate(row); row['strong_success']=strong(row); rows.append(row)
    eligible=[x for x in rows if x['candidate']]
    ranked=sorted(eligible,key=lambda x:(x['fp_boxes'],-x['map'],-x['recall'],-x['f1']))
    pick=ranked[0] if ranked else None
    outcsv=r/'HN2_EPOCH_COMPARISON.csv'
    with outcsv.open('w',newline='',encoding='utf-8-sig') as fobj:
        wr=csv.DictWriter(fobj,fieldnames=list(rows[0].keys()));wr.writeheader();wr.writerows(rows)
    lines=['HN2 RESULT SUMMARY','='*72,f'model: {a.model}',f"baseline validation: mAP={baseline['map']:.6f}, P={baseline['p']:.6f}, R={baseline['r']:.6f}, F1={baseline['f1']:.6f}",f"baseline HN eval60: fp_images={before.get('images_with_fp',baseline['fp_images'])}, fp_boxes={before.get('total_fp',baseline['fp_boxes'])}, max_score={before.get('max_score','?')}",'','EPOCHS']
    for x in rows:
        lines.append(f"e{x['epoch']}: mAP={x['map']:.6f} P={x['precision']:.6f} R={x['recall']:.6f} F1={x['f1']:.6f} | FPimg={x['fp_images']} FPbox={x['fp_boxes']} max={x['max_score']:.6f} | candidate={x['candidate']} strong={x['strong_success']}")
    lines.append('')
    if pick:
        lines += [f"AUTO CANDIDATE: epoch {pick['epoch']}",f"reason: passes fixed {a.model} retention/FP gates; ranked by fewer FP boxes then higher mAP/Recall/F1.","NOTE: this is only the HN2 screening result. Final deployment still requires Pi quantization/live test."]
    else:
        lines += ['AUTO CANDIDATE: NONE','No HN2 epoch passes the pre-fixed retention/FP gates. Keep the original checkpoint for deployment comparison.']
    out=r/'HN2_RESULT_SUMMARY.txt'; out.write_text('\n'.join(lines)+'\n',encoding='utf-8-sig'); print(out.read_text(encoding='utf-8-sig'))

if __name__=='__main__': main()

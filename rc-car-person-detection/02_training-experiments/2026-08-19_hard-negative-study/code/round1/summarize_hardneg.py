from __future__ import annotations
import argparse, csv, math
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--result-dir',type=Path,required=True)
a=p.parse_args()
r=a.result_dir

with (r/'history.csv').open('r',encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
by_epoch={int(float(x['epoch'])):x for x in rows if x.get('epoch')}

def fmt(row,key):
    value=row.get(key,'')
    try: return f"{float(value):.6f}"
    except (TypeError,ValueError): return 'nan'

lines=['HARD NEGATIVE RESULT SUMMARY','='*60,'']
for e in [2,3,5]:
    x=by_epoch.get(e)
    if x:
        lines.append(
            f"epoch {e}: mAP50:95={fmt(x,'metric_map50_95')}, "
            f"P={fmt(x,'metric_precision')}, R={fmt(x,'metric_recall')}, "
            f"F1={fmt(x,'metric_f1')}, valid_total={fmt(x,'valid_total')}"
        )
lines+=['','NEGATIVE EVAL (threshold 0.25)']
for f in sorted((r/'hardneg_eval').glob('*_summary.txt')):
    data={}
    for line in f.read_text(encoding='utf-8-sig').splitlines():
        if ': ' in line:
            k,v=line.split(': ',1); data[k]=v
    lines.append(f"{f.stem}: images_with_fp={data.get('images_with_fp','?')}, total_fp={data.get('total_fp','?')}, max_score={data.get('max_score','?')}")
lines+=['','판정은 e2/e3/e5 중 원본 validation 성능을 최대한 유지하면서 negative FP가 가장 많이 줄어든 checkpoint를 우선한다.']
out=r/'HARDNEG_RESULT_SUMMARY.txt'
out.write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
print(out.read_text(encoding='utf-8-sig'))

from __future__ import annotations
import csv,json,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from common_r46 import package_root,letterbox_image,list_images,detections_from_raw,greedy_match

THRESHOLDS=[0.15,0.20,0.25,0.30,0.35,0.40]

def session(path:Path):
    so=ort.SessionOptions();so.intra_op_num_threads=2;so.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=so,providers=['CPUExecutionProvider'])

def blank_stats():
    return {t:{'reference_detections':0,'candidate_detections':0,'matched':0,'missed':0,'extra':0,'ious':[]} for t in THRESHOLDS}

def grade(s25):
    retain=s25['retain'];iou=s25['mean_iou'];extra=s25['extra_rate_vs_ref']
    if retain>=0.96 and iou>=0.97 and extra<=0.08: return 'PROJECT_STRONG_PASS'
    if retain>=0.95 and iou>=0.95 and extra<=0.10: return 'PASS_FOR_PI_REVIEW'
    if retain>=0.93 and iou>=0.94 and extra<=0.12: return 'CONDITIONAL'
    return 'REVIEW_REQUIRED'

def main():
    pkg=package_root();models=pkg/'models';results=pkg/'results'
    fp=models/'R46_original_448_fp32.onnx'
    candidates={
        'q3q4': models/'R46_original_448_int8_q3_q4.onnx',
        'q2tail4_q3q4': models/'R46_original_448_int8_q2tail4_q3_q4.onnx',
        'q2tail6_q3q4': models/'R46_original_448_int8_q2tail6_q3_q4.onnx',
    }
    images=list_images(pkg/'validation'/'images')
    if len(images)!=120: raise RuntimeError(f'Expected 120 validation images, got {len(images)}')

    ref=session(fp);cand={k:session(v) for k,v in candidates.items()}
    name=ref.get_inputs()[0].name
    x0=letterbox_image(images[0])
    for _ in range(4):
        ref.run(None,{name:x0})
        for s in cand.values(): s.run(None,{name:x0})

    stats={k:blank_stats() for k in candidates}
    lat_ref=[]
    lat={k:[] for k in candidates}
    raw_score={k:[] for k in candidates}
    raw_box={k:[] for k in candidates}
    rows=[]

    print('='*112)
    print('R46 EXTRA LIGHT VALIDATION — FP32 vs Q3+Q4 vs Q2-tail4+Q3+Q4 vs Q2-tail6+Q3+Q4')
    print('='*112)

    for idx,p in enumerate(images,1):
        x=letterbox_image(p)
        t=time.perf_counter(); rb,rs=ref.run(None,{name:x});lat_ref.append((time.perf_counter()-t)*1000)
        rb,rs=rb[0],rs[0]
        row={'image':p.name,'fp32_max_score':float(rs.max())}
        for tag,sess in cand.items():
            t=time.perf_counter(); cb,cs=sess.run(None,{name:x});lat[tag].append((time.perf_counter()-t)*1000)
            cb,cs=cb[0],cs[0]
            raw_score[tag].append(float(np.mean(np.abs(rs-cs))))
            raw_box[tag].append(float(np.mean(np.abs(rb-cb))))
            row[f'{tag}_max_score']=float(cs.max())
            for th in THRESHOLDS:
                rd=detections_from_raw(rb,rs,th,0.5)
                cd=detections_from_raw(cb,cs,th,0.5)
                m,miss,extra=greedy_match(rd,cd,0.5)
                st=stats[tag][th]
                st['reference_detections']+=len(rd)
                st['candidate_detections']+=len(cd)
                st['matched']+=len(m);st['missed']+=miss;st['extra']+=extra
                st['ious'].extend(v[2] for v in m)
                row[f'{tag}_det_{th:.2f}']=len(cd)
        rows.append(row)
        if idx%20==0: print(f'validated {idx}/120')

    reports={}
    for tag in candidates:
        summary={}
        for th,st in stats[tag].items():
            refn=st['reference_detections'];candn=st['candidate_detections'];matched=st['matched'];extra=st['extra']
            summary[f'{th:.2f}']={
                'reference_detections':refn,'candidate_detections':candn,'matched':matched,
                'missed':st['missed'],'extra':extra,
                'retain':matched/refn if refn else 1.0,
                'candidate_ratio':candn/refn if refn else 1.0,
                'extra_rate_vs_ref':extra/refn if refn else 0.0,
                'mean_iou':float(np.mean(st['ious'])) if st['ious'] else 1.0
            }
        s25=summary['0.25']
        reports[tag]={
            'model':str(candidates[tag]),
            'size_mb':candidates[tag].stat().st_size/1024/1024,
            'size_reduction_vs_fp32_percent':100*(1-candidates[tag].stat().st_size/fp.stat().st_size),
            'pc_infer_ms_avg':float(np.mean(lat[tag])),
            'pc_infer_ms_p95':float(np.percentile(lat[tag],95)),
            'raw_score_mean_abs_diff':float(np.mean(raw_score[tag])),
            'raw_box_mean_abs_diff':float(np.mean(raw_box[tag])),
            'detection_agreement':summary,
            'project_grade_at_0_25':grade(s25)
        }

    full={
        'validation_images':120,'threads':2,'nms_iou':0.5,'thresholds':THRESHOLDS,
        'fp32_size_mb':fp.stat().st_size/1024/1024,
        'fp32_pc_infer_ms_avg':float(np.mean(lat_ref)),
        'fp32_pc_infer_ms_p95':float(np.percentile(lat_ref,95)),
        'candidates':reports,
        'decision_rule':'PC latency is NOT the Pi decision metric. Use agreement to reject damaged models, then benchmark survivors on Raspberry Pi.'
    }
    (results/'validation_extra_light_report.json').write_text(
        json.dumps(full,ensure_ascii=False,indent=2),encoding='utf-8'
    )
    with (results/'validation_extra_light_frames.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    print('\nFP32')
    print(' size MB:',round(full['fp32_size_mb'],3),'PC avg/p95:',round(full['fp32_pc_infer_ms_avg'],2),round(full['fp32_pc_infer_ms_p95'],2))
    print('\nCANDIDATE SUMMARY @ conf 0.25 / NMS 0.50')
    print('variant             MB    reduce   ref  cand match miss extra retain    IoU       grade')
    for tag,r in reports.items():
        s=r['detection_agreement']['0.25']
        print(f"{tag:19s} {r['size_mb']:5.3f} {r['size_reduction_vs_fp32_percent']:6.1f}% "
              f"{s['reference_detections']:4d} {s['candidate_detections']:5d} {s['matched']:5d} "
              f"{s['missed']:4d} {s['extra']:5d} {s['retain']*100:6.2f}% {s['mean_iou']:.4f}  {r['project_grade_at_0_25']}")

    print('\nPC CPU LATENCY — reference only, NOT Pi decision')
    for tag,r in reports.items():
        print(f"{tag:19s}: avg {r['pc_infer_ms_avg']:.2f} ms / p95 {r['pc_infer_ms_p95']:.2f} ms")

    print('\nPi test priority:')
    print('1) q3q4 = current control')
    print('2) q2tail4_q3q4 = preferred extra-light candidate')
    print('3) q2tail6_q3q4 = stronger candidate; use only if agreement remains healthy')
    print('saved:',results/'validation_extra_light_report.json')

if __name__=='__main__': main()

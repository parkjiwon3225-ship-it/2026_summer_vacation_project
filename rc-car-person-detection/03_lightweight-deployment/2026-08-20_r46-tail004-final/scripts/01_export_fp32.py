from __future__ import annotations
import json,time
import numpy as np
import torch, onnx, onnxruntime as ort
from common_r46 import *

def main():
    pkg=package_root(); out=pkg/'models'/'R46_original_448_fp32.onnx'; out.parent.mkdir(exist_ok=True)
    model,ckpt,ckpt_path,meta=load_source_model(); wrapper=ExportDetector(model).eval()
    dummy=torch.zeros((1,3,INPUT_H,INPUT_W),dtype=torch.float32)
    with torch.no_grad(): b,s=wrapper(dummy)
    if tuple(b.shape)!=(1,EXPECTED_POINTS,4): raise RuntimeError(f'Unexpected boxes shape: {tuple(b.shape)} expected={(1,EXPECTED_POINTS,4)}')
    if tuple(s.shape)!=(1,EXPECTED_POINTS): raise RuntimeError(f'Unexpected scores shape: {tuple(s.shape)} expected={(1,EXPECTED_POINTS)}')
    print('='*92);print('R46 ORIGINAL FP32 ONNX EXPORT');print('='*92)
    print('checkpoint:',ckpt_path);print('epoch:',ckpt.get('epoch'));print('best_map50_95:',ckpt.get('best_map50_95'))
    print('architecture: fpn_channels=',meta['fpn_channels'],'backbone_expansion=',meta['backbone_expansion'])
    print('input:',(1,3,INPUT_H,INPUT_W));print('boxes:',tuple(b.shape));print('scores:',tuple(s.shape))
    t=time.perf_counter()
    torch.onnx.export(wrapper,dummy,str(out),export_params=True,opset_version=17,do_constant_folding=True,input_names=['images'],output_names=['boxes','scores'],dynamic_axes=None,dynamo=False)
    print('export_sec:',round(time.perf_counter()-t,3))
    onnx.checker.check_model(onnx.load(out))
    so=ort.SessionOptions();so.intra_op_num_threads=2;so.inter_op_num_threads=1
    sess=ort.InferenceSession(str(out),sess_options=so,providers=['CPUExecutionProvider'])
    print('ORT input:',sess.get_inputs()[0].name,sess.get_inputs()[0].shape);print('ORT outputs:',[(o.name,o.shape) for o in sess.get_outputs()])
    samples=list_images(pkg/'validation'/'images')[:3]
    max_sd=max_bd=0.0; mean_sd=[];mean_bd=[]
    for p in samples:
        x=letterbox_image(p)
        with torch.no_grad():pb,ps=wrapper(torch.from_numpy(x))
        ob,os=sess.run(None,{'images':x});bd=np.abs(pb.numpy()-ob);sd=np.abs(ps.numpy()-os)
        max_sd=max(max_sd,float(sd.max()));max_bd=max(max_bd,float(bd.max()));mean_sd.append(float(sd.mean()));mean_bd.append(float(bd.mean()))
    eq={'score_max_abs_diff':max_sd,'score_mean_abs_diff':float(np.mean(mean_sd)),'box_max_abs_diff':max_bd,'box_mean_abs_diff':float(np.mean(mean_bd))}
    print('equivalence:',eq)
    rep={'checkpoint':str(ckpt_path),'epoch':int(ckpt.get('epoch',-1)),'best_map50_95':float(ckpt.get('best_map50_95',float('nan'))),
         'fpn_channels':meta['fpn_channels'],'backbone_expansion':meta['backbone_expansion'],
         'input':[1,3,INPUT_H,INPUT_W],'boxes':[1,EXPECTED_POINTS,4],'scores':[1,EXPECTED_POINTS],
         'size_mb':out.stat().st_size/1024/1024,'equivalence':eq}
    (pkg/'results').mkdir(exist_ok=True)
    (pkg/'results'/'fp32_export_report.json').write_text(json.dumps(rep,indent=2),encoding='utf-8')
    print('[OK]',out,'size_mb=',round(out.stat().st_size/1024/1024,3))
if __name__=='__main__':main()

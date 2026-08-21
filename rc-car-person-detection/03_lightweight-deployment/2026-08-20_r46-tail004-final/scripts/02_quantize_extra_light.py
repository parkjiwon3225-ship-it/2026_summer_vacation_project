from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader,CalibrationMethod,QuantFormat,QuantType,quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from common_r46 import package_root,letterbox_image,list_images

class Reader(CalibrationDataReader):
    def __init__(self,paths,input_name):
        self.paths=list(paths);self.input_name=input_name;self.rewind()
    def rewind(self): self._it=iter(self.paths)
    def get_next(self):
        try: p=next(self._it)
        except StopIteration: return None
        return {self.input_name:letterbox_image(p)}

def is_backbone(name:str)->bool:
    n=(name or '').lower()
    return '/backbone/' in n or 'backbone/' in n

def split_four(seq):
    n=len(seq);cuts=[0,round(n*.25),round(n*.50),round(n*.75),n]
    return [seq[cuts[i]:cuts[i+1]] for i in range(4)]

def check(path:Path):
    onnx.checker.check_model(onnx.load(path))
    so=ort.SessionOptions();so.intra_op_num_threads=2;so.inter_op_num_threads=1
    sess=ort.InferenceSession(str(path),sess_options=so,providers=['CPUExecutionProvider'])
    inp=sess.get_inputs()[0]
    shape=[int(v) if isinstance(v,int) else 1 for v in inp.shape]
    outs=sess.run(None,{inp.name:np.zeros(shape,dtype=np.float32)})
    return inp.shape,[tuple(x.shape) for x in outs]

def quantize_variant(pre:Path,out:Path,paths,nodes,tag:str,results:Path):
    base=ort.InferenceSession(str(pre),providers=['CPUExecutionProvider'])
    input_name=base.get_inputs()[0].name
    reader=Reader(paths,input_name)
    t=time.perf_counter()
    quantize_static(
        model_input=str(pre),
        model_output=str(out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.Percentile,
        calibration_providers=['CPUExecutionProvider'],
        op_types_to_quantize=['Conv'],
        nodes_to_quantize=nodes,
        extra_options={'CalibPercentile':99.99}
    )
    sec=time.perf_counter()-t
    inp,outs=check(out)
    qm=onnx.load(out);ops={}
    for n in qm.graph.node: ops[n.op_type]=ops.get(n.op_type,0)+1
    rep={
        'variant':tag,
        'calibration_samples':len(paths),
        'percentile':99.99,
        'quantized_nodes':len(nodes),
        'size_mb':out.stat().st_size/1024/1024,
        'quant_sec':sec,
        'input_shape':list(inp),
        'output_shapes':[list(x) for x in outs],
        'op_counts':ops
    }
    (results/f'quantize_{tag}.json').write_text(json.dumps(rep,indent=2),encoding='utf-8')
    print(f'[OK] {tag}: {out.name}')
    print('  quantized nodes:',len(nodes),'size_mb:',round(rep['size_mb'],3),'sec:',round(sec,1),
          'Q/DQ:',ops.get('QuantizeLinear',0),ops.get('DequantizeLinear',0))
    return rep

def main():
    pkg=package_root();models=pkg/'models';results=pkg/'results'
    fp32=models/'R46_original_448_fp32.onnx'
    pre=models/'R46_original_448_fp32_preprocessed.onnx'
    selected=list_images(pkg/'calibration'/'images')
    if len(selected)!=96:
        raise RuntimeError(f'Expected 96 calibration images, got {len(selected)}')

    print('='*104)
    print('R46 EXTRA LIGHT SELECTIVE INT8 LADDER')
    print('='*104)
    print('FP32:',fp32)
    print('calibration:',len(selected),'TRAIN images')
    print('method: QDQ / QInt8 / per-channel / Percentile 99.99')

    quant_pre_process(
        input_model_path=str(fp32),
        output_model_path=str(pre),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=True
    )

    m=onnx.load(pre)
    backbone=[n.name for n in m.graph.node if n.op_type=='Conv' and is_backbone(n.name)]
    if not backbone: raise RuntimeError('No backbone Conv nodes found')
    q1,q2,q3,q4=split_four(backbone)

    # Current known-good control, then two deliberate stronger scopes.
    variants={
        'q3q4': q3+q4,
        'q2tail4_q3q4': q2[-min(4,len(q2)):] + q3 + q4,
        'q2tail6_q3q4': q2[-min(6,len(q2)):] + q3 + q4,
    }

    print('backbone Conv:',len(backbone),'quarters:',[len(q1),len(q2),len(q3),len(q4)])
    print('scope ladder:')
    for tag,nodes in variants.items():
        print(f'  {tag:16s}: {len(nodes)}/{len(backbone)} backbone Conv ({100*len(nodes)/len(backbone):.1f}%)')

    with (results/'extra_light_node_groups.txt').open('w',encoding='utf-8') as f:
        for group_name,group in [('Q1',q1),('Q2',q2),('Q3',q3),('Q4',q4)]:
            f.write(f'[{group_name}] {len(group)} nodes\n')
            for x in group: f.write(x+'\n')
            f.write('\n')
        for tag,nodes in variants.items():
            f.write(f'[{tag}] {len(nodes)} quantized nodes\n')
            for x in nodes: f.write(x+'\n')
            f.write('\n')

    reports={}
    outputs={
        'q3q4': models/'R46_original_448_int8_q3_q4.onnx',
        'q2tail4_q3q4': models/'R46_original_448_int8_q2tail4_q3_q4.onnx',
        'q2tail6_q3q4': models/'R46_original_448_int8_q2tail6_q3_q4.onnx',
    }
    for tag,nodes in variants.items():
        reports[tag]=quantize_variant(pre,outputs[tag],selected,nodes,tag,results)

    summary={
        'backbone_conv_count':len(backbone),
        'quarter_sizes':[len(q1),len(q2),len(q3),len(q4)],
        'variants':reports,
        'intent':(
            'Q3+Q4 is the known-good control. Tail4 and Tail6 extend INT8 into the latter part '
            'of Q2 without touching Q1, FPN, or detection head.'
        )
    }
    (results/'quantize_extra_light_summary.json').write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'
    )

if __name__=='__main__': main()

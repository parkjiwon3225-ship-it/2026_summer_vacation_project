from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--init-weights',type=Path,required=True)
    p.add_argument('--snapshot-epochs',default='2,3,5')
    args=p.parse_args()
    root=args.root.resolve()
    sys.path.insert(0,str(root/'src'))
    import rc_detector.training as training
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    snapshots={int(x.strip()) for x in args.snapshot_epochs.split(',') if x.strip()}
    original_save=training.atomic_torch_save
    def snapshot_save(payload,path):
        original_save(payload,path)
        pth=Path(path)
        epoch=int(payload.get('epoch',-1)) if isinstance(payload,dict) else -1
        if pth.name=='last.pt' and epoch in snapshots:
            extra=pth.with_name(f'epoch_{epoch:03d}.pt')
            original_save(payload,extra)
            print(f'[SNAPSHOT] saved: {extra}')
    training.atomic_torch_save=snapshot_save
    result=training.run_training(root,config,resume_path=None,initial_weights_path=args.init_weights.resolve())
    print(result)

if __name__=='__main__':
    main()

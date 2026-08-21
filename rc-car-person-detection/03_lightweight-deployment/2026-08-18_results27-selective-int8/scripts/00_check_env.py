from __future__ import annotations
import sys

print("=" * 72)
print("RESULTS.27 LIGHTWEIGHT ENV CHECK")
print("=" * 72)

mods = ["torch", "numpy", "PIL", "onnx", "onnxruntime"]
failed = False
for name in mods:
    try:
        m = __import__(name)
        ver = getattr(m, "__version__", "(no __version__)")
        print(f"{name:12s}: OK  {ver}")
    except Exception as e:
        failed = True
        print(f"{name:12s}: MISSING/ERROR  {e}")

if failed:
    print("\nInstall ONNX tools in the active rc-person-detector environment:")
    print("python -m pip install onnx==1.22.0 onnxruntime==1.28.0")
    sys.exit(2)

print("\n[OK] environment ready")

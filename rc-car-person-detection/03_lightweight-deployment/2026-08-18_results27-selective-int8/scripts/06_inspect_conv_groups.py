from __future__ import annotations

from collections import Counter
from pathlib import Path
import onnx

from common import package_root


def bucket(name: str) -> str:
    n = (name or "").lower()
    if "/head/" in n or "head/" in n:
        return "head"
    if "/fpn/" in n or "fpn/" in n:
        return "fpn"
    if "/backbone/" in n or "backbone/" in n:
        return "backbone"
    if "stem" in n:
        return "stem"
    return "other"


def main():
    pkg = package_root()
    model_path = pkg / "models" / "results27_640_fp32.onnx"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    model = onnx.load(model_path)
    convs = [node for node in model.graph.node if node.op_type == "Conv"]

    counts = Counter(bucket(node.name) for node in convs)

    print("=" * 110)
    print("RESULTS.27 FP32 ONNX CONV NODE MAP")
    print("=" * 110)
    print("model :", model_path)
    print("Conv  :", len(convs))
    print("groups:", dict(counts))
    print()

    for i, node in enumerate(convs):
        print(f"{i:03d} [{bucket(node.name):8s}] {node.name}")

    report = pkg / "results" / "round3_conv_node_map.txt"
    report.parent.mkdir(exist_ok=True)
    with report.open("w", encoding="utf-8") as f:
        f.write(f"model={model_path}\n")
        f.write(f"total_conv={len(convs)}\n")
        f.write(f"groups={dict(counts)}\n\n")
        for i, node in enumerate(convs):
            f.write(f"{i:03d} [{bucket(node.name):8s}] {node.name}\n")

    print("\nsaved:", report)


if __name__ == "__main__":
    main()

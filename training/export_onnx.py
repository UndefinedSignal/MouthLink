#!/usr/bin/env python3
"""Экспорт чекпойнта в ONNX и сверка выхода torch vs onnxruntime.

ONNX — чтобы крутить модель из C# (ONNX Runtime) в модуле VRCFT, без Python в
рантайме. Ось батча динамическая; H/W фиксируются под твою камеру при экспорте.
"""
import argparse, json
import numpy as np
import torch
from model import model_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    K = len(ck["channels"])
    h, w = ck["height"], ck["width"]
    net = model_from_checkpoint(ck)

    dummy = torch.zeros(1, 1, h, w)
    # dynamo=False — классический TorchScript-экспортёр: не тянет onnxscript.
    torch.onnx.export(
        net, dummy, a.out,
        input_names=["frame"], output_names=["blendshapes"],
        dynamic_axes={"frame": {0: "batch"}, "blendshapes": {0: "batch"}},
        opset_version=a.opset,
        dynamo=False,
    )
    print(f"экспортировано → {a.out}  (arch={ck.get('arch', 'tiny')}, вход 1x1x{h}x{w}, выход {K}: {ck['channels']})")

    # Сверка с onnxruntime на случайных входах.
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        x = rng.random((4, 1, h, w)).astype(np.float32)
        with torch.no_grad():
            yt = net(torch.from_numpy(x)).numpy()
        yo = sess.run(["blendshapes"], {"frame": x})[0]
        md = float(np.max(np.abs(yt - yo)))
        print(f"сверка torch vs onnxruntime: макс. расхождение={md:.2e}", "OK" if md < 1e-4 else "РАСХОДИТСЯ")
    except ImportError:
        print("onnxruntime не установлен — сверку пропустил")


if __name__ == "__main__":
    main()

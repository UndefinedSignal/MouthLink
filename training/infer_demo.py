#!/usr/bin/env python3
"""Прогон ONNX-модели рта через onnxruntime — на синтетическом датасете или на
записи .mlrec. Не требует torch, только onnxruntime + numpy.

На датасете печатает предсказание рядом с истиной (видно, что модель попала).
На записи печатает предсказания по кадрам — так проверяется будущая реальная
модель на реальных кадрах.
"""
import argparse, json, os, struct
import numpy as np


def load_dataset(path, n):
    frames = np.load(os.path.join(path, "frames.npy"))
    labels = np.load(os.path.join(path, "labels.npy"))
    meta = json.load(open(os.path.join(path, "meta.json")))
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return frames[idx], labels[idx], meta["channels"]


def load_mlrec(path, n):
    d = open(path, "rb").read()
    assert d[:4] == b"MLR1", "не .mlrec"
    p = 8
    frames = []
    while p + 4 <= len(d):
        rl = struct.unpack("<I", d[p:p + 4])[0]; p += 4
        rec = d[p:p + rl]; p += rl
        if len(rec) < 26:
            break
        pixfmt = rec[12]
        w, h = struct.unpack("<HH", rec[14:18])
        plen = struct.unpack("<I", rec[22:26])[0]
        payload = rec[26:26 + plen]
        if pixfmt == 0x01 and len(payload) >= w * h:
            frames.append(np.frombuffer(payload[:w * h], np.uint8).reshape(h, w))
    frames = np.array(frames)
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    return frames[idx], None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--data", help="папка синтетического датасета")
    ap.add_argument("--mlrec", help="запись .mlrec")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])

    if a.data:
        frames, labels, channels = load_dataset(a.data, a.n)
    elif a.mlrec:
        frames, labels, channels = load_mlrec(a.mlrec, a.n)
    else:
        raise SystemExit("нужен --data или --mlrec")

    x = (frames.astype(np.float32) / 255.0)[:, None, :, :]
    pred = sess.run(["blendshapes"], {"frame": x})[0]

    hdr = channels if channels else [f"ch{i}" for i in range(pred.shape[1])]
    print("каналы:", hdr)
    for i in range(len(frames)):
        ps = " ".join(f"{v:.3f}" for v in pred[i])
        if labels is not None:
            ts = " ".join(f"{v:.3f}" for v in labels[i])
            print(f"[{i}] pred={ps}   истина={ts}")
        else:
            print(f"[{i}] pred={ps}")


if __name__ == "__main__":
    main()

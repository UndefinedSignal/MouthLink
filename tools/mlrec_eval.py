#!/usr/bin/env python3
r"""Проверка модели по записи с подсказками: что модель выдаёт на каждой позе.

Прогоняет ONNX-модель по кадрам внутри подсказок (как mlrec_pair --face: после settle и до
tail) и сравнивает с целью каталога. Препроцессинг тот же, что в mlrec_pair и mouthlink_live
(полный кадр → --size по высоте, BILINEAR, /255). Лучше всего — на свежей записи, которой не было
в обучении (новая посадка шлема): это то, что увидит модель вживую.

  python tools/mlrec_eval.py --onnx training\runs\face-final\mouth.onnx d:\mouthrecon\f09.mlrec
  python tools/mlrec_eval.py --onnx ... d:\mouthrecon\f09.mlrec --groups cheeks,tongue

Для каждой позы: по каждому каналу из цели — цель, сток (среднее), модель (среднее «сырое»,
до --lo/--hi) и «попаданий» (доля кадров, где модель по ту же сторону от 0.5, что и цель).
Ниже — какие ещё каналы модель «зажигает» на этой позе (≥ 0.3), и ложные срабатывания новых
каналов на «Расслабь лицо».

Опции:
  --catalog FILE   каталог (по умолчанию prompts/face.json рядом с репозиторием)
  --groups g,g     только позы этих групп
  --csv FILE       сохранить таблицу (поза, канал, цель, сток, модель, попаданий, кадров)
  --cpu            только CPU
"""
import argparse, bisect, csv, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlrec_pair import MOUTH_NAMES, read_mlrec, read_prompts, read_stock, to_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--meta")
    ap.add_argument("--catalog")
    ap.add_argument("--groups")
    ap.add_argument("--csv")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    import onnxruntime as ort
    from PIL import Image

    meta_path = a.meta or os.path.splitext(a.onnx)[0] + ".json"
    if not os.path.exists(meta_path):
        raise SystemExit(f"нет {meta_path} (описание каналов от export_onnx.py); укажи --meta")
    channels = json.load(open(meta_path, encoding="utf-8"))["channels"]
    ci = {c: i for i, c in enumerate(channels)}
    cat_path = a.catalog or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts", "face.json")
    cat = json.load(open(cat_path, encoding="utf-8"))
    dflt = cat.get("defaults", {})
    groups = set(a.groups.split(",")) if a.groups else None
    items = {}
    for it in cat["items"]:
        if groups and it.get("group") not in groups:
            continue
        items[it["id"]] = it
    rest_id = cat.get("rest", {}).get("id", "neutral")
    new_idx = [i for i, c in enumerate(channels) if c not in MOUTH_NAMES]

    avail = ort.get_available_providers()
    want = ["CPUExecutionProvider"] if a.cpu else [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider",
                                                              "CPUExecutionProvider") if p in avail]
    sess = ort.InferenceSession(a.onnx, providers=want)
    inp = sess.get_inputs()[0]
    H, W = int(inp.shape[2]), int(inp.shape[3])

    paths = []
    for p in a.sessions:
        paths += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    # поза → накопители: сумма выхода модели [K], попадания по каналам цели, сумма стока по каналам цели, кадров
    acc = {}
    rest_sum = np.zeros(len(channels)); rest_hits = np.zeros(len(channels)); rest_n = 0

    def flush(batch, keys, stocks):
        nonlocal rest_n
        if not batch:
            return
        x = (np.stack(batch).astype(np.float32) / 255.0)[:, None]
        y = np.clip(sess.run(None, {inp.name: x})[0], 0, 1)
        for row, key, sv in zip(y, keys, stocks):
            if key == rest_id:
                rest_sum[:] += row; rest_hits[:] += row >= 0.5; rest_n += 1
                continue
            r = acc.setdefault(key, {"sum": np.zeros(len(channels)), "hit": {}, "stock": {}, "n": 0})
            r["sum"] += row; r["n"] += 1
            for ch, target in items[key].get("labels", {}).items():
                if ch not in ci:
                    continue
                r["hit"][ch] = r["hit"].get(ch, 0) + int((row[ci[ch]] >= 0.5) == (target >= 0.5))
                if ch in MOUTH_NAMES and sv is not None:
                    r["stock"][ch] = r["stock"].get(ch, 0.0) + float(np.clip(sv[MOUTH_NAMES.index(ch)], 0, 1))
        batch.clear(); keys.clear(); stocks.clear()

    for path in paths:
        prompts = read_prompts(os.path.splitext(path)[0] + ".prompts.csv")
        if not prompts:
            print(f"пропускаю {path}: нет .prompts.csv")
            continue
        sp = os.path.splitext(path)[0] + ".stock.csv"
        stock = read_stock(sp, "instant") if os.path.exists(sp) else {}
        starts = [p[0] for p in prompts]
        batch, keys, stocks = [], [], []
        n = 0
        for fid, cap, pf, w, h, rx, ry, payload in read_mlrec(path):
            j = bisect.bisect_right(starts, cap) - 1
            if j < 0 or cap >= prompts[j][1]:
                continue
            s0, e0, lab = prompts[j]
            it = items.get(lab)
            if lab == rest_id:
                settle, tail = dflt.get("settle", 0.8), dflt.get("tail", 0.2)
            elif it is not None:
                settle, tail = it.get("settle", dflt.get("settle", 0.8)), it.get("tail", dflt.get("tail", 0.2))
            else:
                continue
            if not (s0 + settle * 1e9 <= cap <= e0 - tail * 1e9):
                continue
            img = to_image(pf, w, h, payload)
            if img.shape != (H, W):
                img = np.asarray(Image.fromarray(img).resize((W, H), Image.BILINEAR))
            row = stock.get(cap)
            batch.append(img); keys.append(lab); stocks.append(row[2] if row else None); n += 1
            if len(batch) >= a.batch:
                flush(batch, keys, stocks)
        flush(batch, keys, stocks)
        print(f"{path}: кадров в подсказках {n}")

    if not acc:
        raise SystemExit("ни одной позы из каталога в записях")
    rows = []
    order = [i for i in items if i in acc]
    print(f"\nмодель {a.onnx}; «модель» — сырой выход (до --lo/--hi), попаданий — по порогу 0.5\n")
    print(f"{'поза':<18}{'канал':<20}{'цель':>5}{'сток':>7}{'модель':>8}{'попаданий':>11}")
    worst = []
    for key in order:
        r = acc[key]; n = r["n"]; mean = r["sum"] / n
        labels = items[key].get("labels", {})
        for ch, target in labels.items():
            if ch not in ci:
                continue
            st = r["stock"].get(ch)
            st_s = f"{st / n:7.2f}" if st is not None else "      —"
            hit = r["hit"].get(ch, 0) / n
            print(f"{key:<18}{ch:<20}{target:5.1f}{st_s}{mean[ci[ch]]:8.2f}{hit:10.0%}")
            rows.append([key, ch, target, "" if st is None else round(st / n, 3), round(float(mean[ci[ch]]), 3),
                         round(hit, 3), n])
            if target >= 0.5:
                worst.append((hit, key, ch))
        others = [(mean[i], channels[i]) for i in np.argsort(-mean) if mean[i] >= 0.3 and channels[i] not in labels]
        if others:
            print(f"{'':<18}ещё: " + ", ".join(f"{c} {v:.2f}" for v, c in others[:5]))
    if rest_n:
        fp = [(rest_hits[i] / rest_n, channels[i]) for i in new_idx if rest_hits[i] / rest_n > 0.01]
        print(f"\n«{rest_id}» ({rest_n} кадров): ложные срабатывания новых каналов "
              + (", ".join(f"{c} {v:.0%}" for v, c in sorted(fp, reverse=True)) if fp else "нет (≤ 1 %)"))
    worst.sort()
    bad = [f"{k}/{c} {h:.0%}" for h, k, c in worst if h < 0.6]
    print("\nслабые места (цель ≥ 0.5, попаданий < 60 %): " + (", ".join(bad) if bad else "нет"))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", "channel", "target", "stock_mean", "model_mean", "hit_rate", "frames"])
            w.writerows(rows)
        print(f"таблица → {a.csv}")


if __name__ == "__main__":
    main()

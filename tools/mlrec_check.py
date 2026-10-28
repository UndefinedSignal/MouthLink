#!/usr/bin/env python3
"""Проверка записи .mlrec: дошло ли изображение и здоров ли поток.

Читает запись потоково (годится для многогигабайтных файлов) и сообщает:
  * частоту и ровность кадров по меткам времени (медиана/p95/макс интервала);
  * пропуски и повторы номеров кадров, нарушение порядка, немонотонное время;
  * смену размера/формата посреди записи;
  * «плохие» кадры: почти чёрные, пересвеченные, без контраста;
  * застывшую камеру: подряд идущие побайтно одинаковые кадры;
  * с --pattern: попиксельную сверку с тестовым шаблоном (эмулятор --pattern).

В конце — вердикт и список замечаний. --csv сохраняет статистику по каждому кадру.

    python mlrec_check.py session.mlrec [--pattern] [--csv frames.csv]

Кадры не сохраняются и не показываются — только числа.
"""
import argparse, csv, hashlib, struct, sys
import numpy as np

HDR_FMT = "<IQBBHHHHI"           # frameId, captureNs, pixfmt, comp, w, h, roiX, roiY, payloadLen
HDR_SIZE = struct.calcsize(HDR_FMT)   # 26
FMT_NAMES = {0x01: "GRAY8", 0x02: "GRAY16LE", 0x10: "JPEG"}


def pattern(w, h, fid):
    """Та же формула, что TestPattern.Value в MouthLink.Core."""
    y, x = np.mgrid[0:h, 0:w].astype(np.int64)
    return ((7 * x + 13 * y + 31 * (fid & 0xFFFF) + ((x * y) >> 4)) & 0xFF).astype(np.uint8)


def iter_records(path):
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8 or head[:4] != b"MLR1":
            sys.exit("не .mlrec (нет магии MLR1)")
        while True:
            lb = f.read(4)
            if len(lb) < 4:
                return
            rl = struct.unpack("<I", lb)[0]
            rec = f.read(rl)
            if len(rec) < rl:
                yield None            # обрезанный хвост
                return
            if rl < HDR_SIZE:
                yield None
                return
            fid, cap, pf, comp, w, h, rx, ry, plen = struct.unpack(HDR_FMT, rec[:HDR_SIZE])
            yield dict(fid=fid, t=cap, pf=pf, comp=comp, w=w, h=h, plen=plen, payload=rec[HDR_SIZE:HDR_SIZE + plen])


def frame_stats(r):
    """Статистика пикселей для несжатых монохромных кадров (иначе None)."""
    w, h, pf, comp, p = r["w"], r["h"], r["pf"], r["comp"], r["payload"]
    if comp != 0:
        return None
    if pf == 0x01 and len(p) == w * h:
        a = np.frombuffer(p, np.uint8).reshape(h, w).astype(np.float32)
        lo, hi = 2, 253
    elif pf == 0x02 and len(p) == w * h * 2:
        a = np.frombuffer(p, "<u2").reshape(h, w).astype(np.float32)
        top = float(a.max()) if a.size else 1.0
        lo, hi = 2, max(top * 0.99, 3)
    else:
        return None
    return dict(mean=float(a.mean()), std=float(a.std()),
                black=float((a <= lo).mean()), sat=float((a >= hi).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--pattern", action="store_true", help="сверять с тестовым шаблоном эмулятора")
    ap.add_argument("--csv", help="сохранить статистику по кадрам")
    a = ap.parse_args()

    n = 0
    truncated = False
    ids, ts = [], []
    shapes = {}
    shape_changes = 0
    prev_shape = None
    bad_size = 0
    black = sat = flat = 0
    frozen_run = longest_frozen = frozen_frames = 0
    prev_hash = None
    pat_checked = pat_bad = pat_worst = 0
    rows = []

    for r in iter_records(a.rec):
        if r is None:
            truncated = True
            break
        n += 1
        ids.append(r["fid"]); ts.append(r["t"])
        shape = (r["w"], r["h"], r["pf"], r["comp"])
        shapes[shape] = shapes.get(shape, 0) + 1
        if prev_shape is not None and shape != prev_shape:
            shape_changes += 1
        prev_shape = shape
        bpp = 2 if r["pf"] == 0x02 else 1
        if r["comp"] == 0 and r["pf"] in (0x01, 0x02) and len(r["payload"]) != r["w"] * r["h"] * bpp:
            bad_size += 1

        hsh = hashlib.blake2b(r["payload"], digest_size=8).digest()
        if hsh == prev_hash:
            frozen_run += 1
            frozen_frames += 1
            longest_frozen = max(longest_frozen, frozen_run + 1)
        else:
            frozen_run = 0
        prev_hash = hsh

        st = frame_stats(r)
        if st:
            if st["black"] > 0.98: black += 1
            if st["sat"] > 0.98: sat += 1
            if st["std"] < 2.0: flat += 1

        mism = ""
        if a.pattern:
            pat_checked += 1
            if r["pf"] == 0x01 and r["comp"] == 0 and len(r["payload"]) == r["w"] * r["h"]:
                got = np.frombuffer(r["payload"], np.uint8).reshape(r["h"], r["w"])
                mism = int((got != pattern(r["w"], r["h"], r["fid"])).sum())
            else:
                mism = r["w"] * r["h"]
            if mism:
                pat_bad += 1
                pat_worst = max(pat_worst, mism)

        if a.csv:
            pix = [f"{st[k]:.3f}" for k in ("mean", "std", "black", "sat")] if st else ["", "", "", ""]
            rows.append([r["fid"], r["t"], r["w"], r["h"], FMT_NAMES.get(r["pf"], hex(r["pf"])), r["comp"], *pix, mism])

    if n == 0:
        sys.exit("в записи нет кадров")

    ids = np.array(ids, np.int64)
    ts = np.array(ts, np.float64)
    did = np.diff(ids)
    gaps = int(did[did > 1].sum() - (did > 1).sum()) if len(did) else 0
    dup = int((did == 0).sum())
    back = int((did < 0).sum())
    dt = np.diff(ts) / 1e6
    nonmono = int((dt <= 0).sum())
    dur = (ts[-1] - ts[0]) / 1e9 if n > 1 else 0.0
    fps = (n - 1) / dur if dur > 0 else 0.0
    good_dt = dt[dt > 0]

    print(f"запись: {a.rec}")
    print(f"кадров: {n}, длительность {dur:.2f} с, средняя частота {fps:.1f} кадр/с"
          + ("  (ХВОСТ ОБРЕЗАН)" if truncated else ""))
    for (w, h, pf, comp), cnt in shapes.items():
        print(f"  формат: {w}x{h} {FMT_NAMES.get(pf, hex(pf))} сжатие={comp} — {cnt} кадров")
    if len(good_dt):
        p50, p95, mx = np.percentile(good_dt, 50), np.percentile(good_dt, 95), good_dt.max()
        print(f"интервал между кадрами: медиана {p50:.2f} мс, p95 {p95:.2f} мс, макс {mx:.2f} мс")
    print(f"номера кадров: пропущено {gaps}, повторов {dup}, назад {back}; немонотонное время: {nonmono}")
    print(f"плохие кадры: чёрных {black}, пересвеченных {sat}, без контраста {flat}; "
          f"неверный размер payload {bad_size}")
    print(f"застывшие кадры: {frozen_frames} (самая длинная серия одинаковых: {longest_frozen})")
    if a.pattern:
        print(f"тестовый шаблон: {pat_checked - pat_bad}/{pat_checked} кадров совпали попиксельно"
              + (f", худший кадр — {pat_worst} пикс." if pat_bad else ""))

    issues = []
    if truncated: issues.append("запись обрезана (приёмник остановлен посреди записи кадра)")
    if gaps: issues.append(f"пропущено {gaps} номеров кадров ({100 * gaps / (n + gaps):.1f}%)")
    if dup or back: issues.append("повторы/нарушение порядка номеров кадров")
    if nonmono: issues.append("метки времени не монотонны — проверь источник времени кадра")
    if shape_changes: issues.append(f"размер/формат менялся {shape_changes} раз")
    if bad_size: issues.append("payload не совпадает с размером кадра")
    if black > 0.05 * n: issues.append("много чёрных кадров — камера выключена или не тот буфер?")
    if sat > 0.05 * n: issues.append("много пересвеченных кадров")
    if flat > 0.05 * n: issues.append("много кадров без контраста — не изображение?")
    if longest_frozen >= 5: issues.append(f"камера застывала: {longest_frozen} одинаковых кадров подряд")
    if len(good_dt) and np.percentile(good_dt, 95) > 2.5 * np.percentile(good_dt, 50):
        issues.append("неровный темп кадров (p95 интервала > 2.5× медианы)")
    if a.pattern and pat_bad: issues.append(f"{pat_bad} кадров не совпали с тестовым шаблоном")

    print("ВЕРДИКТ:", "OK" if not issues else "ЕСТЬ ЗАМЕЧАНИЯ")
    for s in issues:
        print("  -", s)

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame_id", "capture_ns", "w", "h", "fmt", "comp", "mean", "std", "black", "sat", "pattern_mismatch"])
            wr.writerows(rows)
        print("статистика по кадрам →", a.csv)
    sys.exit(0 if not issues else 2)


if __name__ == "__main__":
    main()

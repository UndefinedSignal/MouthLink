#!/usr/bin/env python3
r"""Спаривание кадров MouthLink со стоковыми значениями рта → датасет для training/train.py.

Каждая сессия — запись приёмника с --record: <имя>.mlrec (кадры) и <имя>.stock.csv
(пакеты STOCK от модуля ml3+). Кадр и стоковые значения спариваются по ТОЧНОМУ
равенству метки времени источника (capture_ns кадра == source_ns пакета).

  python tools/mlrec_pair.py OUT s03.mlrec s04.mlrec [опции]
  python tools/mlrec_pair.py OUT d:\mouthrecon\s*.mlrec [опции]

Шаблон с * раскрывается сам (PowerShell этого не делает); записи из шаблона без
.stock.csv пропускаются с предупреждением, явно перечисленные — ошибка.

Опции:
  --labels instant|final   instant (по умолч.) — после стоковой постобработки без
                           временного сглаживания: метка без запаздывания относительно
                           кадра. final — сглаженные значения, как в VRCFT (запаздывают).
  --crop x,y,w,h           вырезать прямоугольник (координаты полного кадра 400×400)
  --size N                 масштабировать: высота N, ширина по пропорции (нужен Pillow)
  --every K                брать каждый K-й спаренный кадр (90 Гц очень избыточны)
  --keep-invalid           не выкидывать кадры, где сток считает рот невалидным
  --keep-warmup            не выкидывать кадры прогрева фильтра стока
  --tongue                 добавить каналы направления языка TongueLeft/Right/Up/Down:
                           внутри подсказок из <имя>.prompts.csv (приёмник --prompts tongue)
                           — по подсказке; вне подсказок 0, если сток говорит «язык
                           не высунут» (TongueOut < --neg-tongue), иначе NaN (неизвестно).
                           Записи без подсказок дают только такие нули — тоже полезно.
  --settle 0.8 --tail 0.2  сколько секунд отрезать в начале (реакция) и в конце подсказки
  --neg-tongue 0.05        порог стокового TongueOut для «направления нет»
  --face                   сценарий выражений (приёмник --prompts face): каналы 33 стока +
                           TongueLeft/Right/Up/Down + новые каналы каталога (щёки раздельно,
                           втянутые щёки, напряжённые губы, язык трубочкой). Внутри выражения
                           (после его settle, без tail) каналы из каталога берутся из сценария,
                           остальные стоковые — от стока; вне выражений: сток + автоправила
                           (язык не высунут → направления и TongueRoll = 0; щёки не надуты →
                           CheekPuffLeft/Right = 0), прочие новые каналы неизвестны (NaN).
                           Включает и разметку языка из записей --prompts tongue.
  --catalog FILE           каталог выражений (по умолчанию ../prompts/face.json от скрипта)
  --neg-cheek 0.05         порог стокового CheekPuff для «щёки не надуты»
  --flip                   записать правила отражения L↔R в meta.json (включает
                           отражение в аугментациях). Только если камера рта стоит по
                           центру лица: иначе отражённый кадр — ракурс, которого нет.

Выход в OUT: frames.npy uint8 [N,H,W], labels.npy float32 [N,33], groups.npy [N]
(номер сессии = номер файла в списке), meta.json, pair-report.json.
"""
import argparse, bisect, csv, glob, json, os, struct, sys

import numpy as np

# Порядок mouth33 и имена слотов PICO — как в MouthLink.Core/StockPacket.cs.
MOUTH_SLOTS = [43, 29, 37, 25, 32, 7, 15, 5, 49, 23, 27, 40, 10, 24, 50, 6, 20,
               22, 34, 48, 18, 14, 17, 9, 19, 21, 33, 42, 39, 8, 1, 13, 51]
PICO_SHAPES = [
    "EyeLookDownL", "NoseSneerL", "EyeLookInL", "BrowInnerUp", "BrowDownR", "MouthClose",
    "MouthLowerDownR", "JawOpen", "MouthUpperUpR", "MouthShrugUpper", "MouthFunnel",
    "EyeLookInR", "EyeLookDownR", "NoseSneerR", "MouthRollUpper", "JawRight", "BrowDownL",
    "MouthShrugLower", "MouthRollLower", "MouthSmileL", "MouthPressL", "MouthSmileR",
    "MouthPressR", "MouthDimpleR", "MouthLeft", "JawForward", "EyeSquintL", "MouthFrownL",
    "EyeBlinkL", "CheekSquintL", "BrowOuterUpL", "EyeLookUpL", "JawLeft", "MouthStretchL",
    "MouthPucker", "EyeLookUpR", "BrowOuterUpR", "CheekSquintR", "EyeBlinkR", "MouthUpperUpL",
    "MouthFrownR", "EyeSquintR", "MouthStretchR", "CheekPuff", "EyeLookOutL", "EyeLookOutR",
    "EyeWideR", "EyeWideL", "MouthRight", "MouthDimpleL", "MouthLowerDownL", "TongueOut",
]
MOUTH_NAMES = [PICO_SHAPES[s] for s in MOUTH_SLOTS]
TONGUE_DIRS = ["TongueLeft", "TongueRight", "TongueUp", "TongueDown"]
# Метки подсказок (MouthLink.Receiver/Prompter.cs) → значения TONGUE_DIRS.
PROMPT_DIRS = {
    "neutral": (0, 0, 0, 0),
    "tongue_out": (0, 0, 0, 0),      # прямо: высунут, но без направления
    "tongue_left": (1, 0, 0, 0),
    "tongue_right": (0, 1, 0, 0),
    "tongue_up": (0, 0, 1, 0),
    "tongue_down": (0, 0, 0, 1),
}


def load_catalog(path, channels):
    """id → (список (индекс канала, значение), settle_ns, tail_ns, zero_new, индексы unknown).

    unknown — каналы, которые в этой подсказке размечать нельзя (NaN): например, направления
    языка, пока его медленно водят из стороны в сторону."""
    c = json.load(open(path, encoding="utf-8"))
    d = c.get("defaults", {})
    ci = {ch: i for i, ch in enumerate(channels)}
    specs = {}
    rest = c.get("rest", {})
    specs[rest.get("id", "neutral")] = ([], int(d.get("settle", 0.8) * 1e9), int(d.get("tail", 0.2) * 1e9), True, [])
    for it in c["items"]:
        bad = [k for k in list(it.get("labels", {})) + list(it.get("unknown", [])) if k not in ci]
        if bad:
            raise SystemExit(f"{path}: в выражении {it['id']} неизвестные каналы {bad}")
        specs[it["id"]] = ([(ci[k], float(v)) for k, v in it.get("labels", {}).items()],
                           int(it.get("settle", d.get("settle", 0.8)) * 1e9),
                           int(it.get("tail", d.get("tail", 0.2)) * 1e9),
                           bool(it.get("zero_new", d.get("zero_new", True))),
                           [ci[k] for k in it.get("unknown", [])])
    return c, specs


def read_prompts(path):
    """[(start_ns, end_ns, label)] по возрастанию start; нет файла → []."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append((int(row["start_ns"]), int(row["end_ns"]), row["label"]))
    out.sort()
    return out


def flip_pairs(names):
    pairs, seen = [], set()
    for n in names:
        for a, b in (("Left", "Right"), ("L", "R")):
            if n.endswith(a):
                m = n[: -len(a)] + b
                if m in names and n not in seen:
                    pairs.append([n, m]); seen.update((n, m))
    return pairs


def read_mlrec(path):
    """Итератор кадров: (frame_id, capture_ns, pixfmt, w, h, roi_x, roi_y, payload bytes)."""
    with open(path, "rb") as f:
        if f.read(4) != b"MLR1":
            raise SystemExit(f"{path}: не .mlrec")
        f.read(4)
        while True:
            head = f.read(4)
            if len(head) < 4:
                return
            (L,) = struct.unpack("<I", head)
            rec = f.read(L)
            if len(rec) < L:
                return  # обрезанный хвост старых записей
            fid, cap, pf, comp, w, h, rx, ry, pl = struct.unpack_from("<IQBBHHHHI", rec, 0)
            if comp != 0:
                raise SystemExit(f"{path}: сжатые кадры пока не поддержаны")
            yield fid, cap, pf, w, h, rx, ry, rec[26:26 + pl]


def read_stock(path, kind):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        hdr = next(r)
        col = {n: i for i, n in enumerate(hdr)}
        prefix = "inst_" if kind == "instant" else "fin_"
        idx = [col[prefix + n] for n in MOUTH_NAMES]
        for row in r:
            if not row:
                continue
            ns = int(row[0])
            rows[ns] = (row[2] == "1", row[3] == "1",
                        np.array([float(row[i]) for i in idx], np.float32))
    return rows


def to_image(pf, w, h, payload):
    if pf == 1:
        return np.frombuffer(payload, np.uint8).reshape(h, w)
    if pf == 2:
        return (np.frombuffer(payload, "<u2").reshape(h, w) >> 8).astype(np.uint8)
    raise SystemExit(f"формат пикселей {pf} не поддержан")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--labels", choices=["instant", "final"], default="instant")
    ap.add_argument("--crop")
    ap.add_argument("--size", type=int)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--keep-invalid", action="store_true")
    ap.add_argument("--keep-warmup", action="store_true")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--tongue", action="store_true")
    ap.add_argument("--settle", type=float, default=0.8)
    ap.add_argument("--tail", type=float, default=0.2)
    ap.add_argument("--neg-tongue", type=float, default=0.05)
    ap.add_argument("--face", action="store_true")
    ap.add_argument("--catalog")
    ap.add_argument("--neg-cheek", type=float, default=0.05)
    a = ap.parse_args()

    crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    if crop and (len(crop) != 4 or crop[2] <= 0 or crop[3] <= 0):
        raise SystemExit("--crop ожидает x,y,w,h")
    resize = None
    if a.size:
        try:
            from PIL import Image
        except ImportError:
            raise SystemExit("--size требует Pillow: pip install pillow")
        resize = Image

    sessions = []
    for pattern in a.sessions:
        if any(c in pattern for c in "*?["):
            found = sorted(glob.glob(pattern))
            if not found:
                raise SystemExit(f"по шаблону {pattern} ничего не найдено")
            for path in found:
                if os.path.exists(os.path.splitext(path)[0] + ".stock.csv"):
                    sessions.append(path)
                else:
                    print(f"пропускаю {path}: нет .stock.csv (запись без стока)")
        else:
            sessions.append(pattern)
    if not sessions:
        raise SystemExit("нет ни одной записи со стоком")
    a.sessions = sessions

    frames, labels, groups, report = [], [], [], {"labels": a.labels, "sessions": []}
    out_hw = None
    channels = MOUTH_NAMES + (TONGUE_DIRS if a.tongue else [])
    i_tongue_out = MOUTH_NAMES.index("TongueOut")
    face_specs = None
    if a.face:
        a.tongue = False   # своя ветка: язык размечается по каталогу (метки tongue_* совпадают)
        cat_path = a.catalog or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts", "face.json")
        if not os.path.exists(cat_path):
            raise SystemExit(f"нет каталога {cat_path}; укажи --catalog")
        cat = json.load(open(cat_path, encoding="utf-8"))
        new_ch = cat["new_channels"]
        channels = MOUTH_NAMES + TONGUE_DIRS + new_ch
        cat, face_specs = load_catalog(cat_path, channels)
        ci = {ch: i for i, ch in enumerate(channels)}
        i_dirs = [ci[d] for d in TONGUE_DIRS]
        i_roll = ci.get("TongueRoll")
        i_puff_lr = [ci[x] for x in ("CheekPuffLeft", "CheekPuffRight") if x in ci]
        i_cheek = MOUTH_NAMES.index("CheekPuff")
        zero_idx = [ci[x] for x in TONGUE_DIRS + new_ch]
        item_stats = {}      # id → {канал: [сумма стока, кадров, цель]}
        unknown_labels = set()
        print(f"каталог: {os.path.abspath(cat_path)} ({len(cat['items'])} выражений, новых каналов {len(new_ch)})")
    settle_ns, tail_ns = int(a.settle * 1e9), int(a.tail * 1e9)
    for gi, sess in enumerate(a.sessions):
        stock_path = os.path.splitext(sess)[0] + ".stock.csv"
        if not os.path.exists(stock_path):
            raise SystemExit(f"нет {stock_path}: запись сделана без стока (нужен модуль ml3 и приёмник без --no-stock)")
        stock = read_stock(stock_path, a.labels)
        st = dict(file=sess, stock_rows=len(stock), frames=0, matched=0, no_stock=0,
                  invalid=0, warmup=0, taken=0)
        prompts = read_prompts(os.path.splitext(sess)[0] + ".prompts.csv") if (a.tongue or a.face) else []
        starts = [p[0] for p in prompts]
        if a.face:
            st["prompts"] = len(prompts)
        if a.tongue:
            st["prompts"] = len(prompts)
            st["tongue"] = {"positive": {d: 0 for d in TONGUE_DIRS}, "zero": 0, "unknown": 0,
                            "stock_tongue_out_by_prompt": {}}
            tout_sum, tout_cnt = {}, {}
        used = set()
        k = 0
        for fid, cap, pf, w, h, rx, ry, payload in read_mlrec(sess):
            st["frames"] += 1
            row = stock.get(cap)
            if row is None:
                st["no_stock"] += 1
                continue
            used.add(cap)
            st["matched"] += 1
            valid, warm, vec = row
            if not valid and not a.keep_invalid:
                st["invalid"] += 1
                continue
            if warm and not a.keep_warmup:
                st["warmup"] += 1
                continue
            dirs = None
            in_prompt = False
            if a.tongue:
                j = bisect.bisect_right(starts, cap) - 1
                if j >= 0 and cap < prompts[j][1]:
                    s0, e0, lab = prompts[j]
                    tout_sum[lab] = tout_sum.get(lab, 0.0) + float(vec[i_tongue_out])
                    tout_cnt[lab] = tout_cnt.get(lab, 0) + 1
                    if lab in PROMPT_DIRS and s0 + settle_ns <= cap <= e0 - tail_ns:
                        dirs = np.asarray(PROMPT_DIRS[lab], np.float32)
                        in_prompt = True
                if dirs is None:
                    dirs = (np.zeros(4, np.float32) if vec[i_tongue_out] < a.neg_tongue
                            else np.full(4, np.nan, np.float32))
            face_vec = None
            if a.face:
                face_vec = np.full(len(channels), np.nan, np.float32)
                face_vec[:len(MOUTH_NAMES)] = np.clip(vec, 0.0, 1.0)
                if vec[i_tongue_out] < a.neg_tongue:
                    face_vec[i_dirs] = 0
                    if i_roll is not None:
                        face_vec[i_roll] = 0
                if vec[i_cheek] < a.neg_cheek:
                    face_vec[i_puff_lr] = 0
                j = bisect.bisect_right(starts, cap) - 1
                if j >= 0 and cap < prompts[j][1]:
                    s0, e0, lab = prompts[j]
                    spec = face_specs.get(lab)
                    if spec is None:
                        unknown_labels.add(lab)
                    else:
                        pairs, settle, tail, zero_new, unknown = spec
                        if s0 + settle <= cap <= e0 - tail:
                            if zero_new:
                                face_vec[zero_idx] = 0
                            for idx, val in pairs:
                                if idx < len(MOUTH_NAMES):
                                    rec = item_stats.setdefault(lab, {}).setdefault(channels[idx], [0.0, 0, val])
                                    rec[0] += float(np.clip(vec[idx], 0, 1)); rec[1] += 1
                                face_vec[idx] = val
                            if unknown:
                                face_vec[unknown] = np.nan
                            in_prompt = True
            k += 1
            # Кадры внутри подсказок редкие и ценные: прореживание --every их не трогает.
            if (k - 1) % a.every and not in_prompt:
                continue
            img = to_image(pf, w, h, payload)
            if crop:
                x, y, cw, ch = crop
                x -= rx; y -= ry
                if x < 0 or y < 0 or x + cw > w or y + ch > h:
                    raise SystemExit(f"{sess}: --crop {a.crop} не помещается в кадр {w}x{h}+{rx},{ry}")
                img = img[y:y + ch, x:x + cw]
            if resize:
                th = a.size
                tw = max(1, round(img.shape[1] * th / img.shape[0]))
                img = np.asarray(resize.fromarray(img).resize((tw, th), resize.BILINEAR))
            if out_hw is None:
                out_hw = img.shape
            elif img.shape != out_hw:
                raise SystemExit(f"{sess}: размер кадра {img.shape} ≠ {out_hw}; задай одинаковые --crop/--size")
            frames.append(np.ascontiguousarray(img))
            lab_vec = face_vec if a.face else np.clip(vec, 0.0, 1.0)
            if a.tongue:
                lab_vec = np.concatenate([lab_vec, dirs])
                if np.isnan(dirs).any():
                    st["tongue"]["unknown"] += 1
                elif dirs.any():
                    st["tongue"]["positive"][TONGUE_DIRS[int(np.argmax(dirs))]] += 1
                else:
                    st["tongue"]["zero"] += 1
            labels.append(lab_vec)
            groups.append(gi)
            st["taken"] += 1
        st["stock_without_frame"] = len(stock) - len(used)
        st["match_rate"] = round(st["matched"] / st["frames"], 4) if st["frames"] else 0.0
        if a.tongue:
            st["tongue"]["stock_tongue_out_by_prompt"] = {
                lab: round(tout_sum[lab] / tout_cnt[lab], 3) for lab in sorted(tout_sum)}
        report["sessions"].append(st)
        print(f"{sess}: кадров {st['frames']}, стока {st['stock_rows']}, спарено {st['matched']} "
              f"({100 * st['match_rate']:.1f}%), без стока {st['no_stock']}, сток без кадра "
              f"{st['stock_without_frame']}, невалидных {st['invalid']}, прогрев {st['warmup']} → взято {st['taken']}")
        if a.tongue:
            t = st["tongue"]
            pos = ", ".join(f"{d[6:]}={n}" for d, n in t["positive"].items())
            print(f"    язык: подсказок {len(prompts)}; направления: {pos}; нулей {t['zero']}, неизвестно {t['unknown']}")
            if t["stock_tongue_out_by_prompt"]:
                print("    стоковый TongueOut по подсказкам: " +
                      ", ".join(f"{k}={v}" for k, v in t["stock_tongue_out_by_prompt"].items()))

    if not frames:
        raise SystemExit("ни одной пары: проверь, что запись сделана с модулем ml3")
    os.makedirs(a.out, exist_ok=True)
    F = np.stack(frames)
    Y = np.stack(labels).astype(np.float32)
    G = np.asarray(groups, np.int32)
    np.save(os.path.join(a.out, "frames.npy"), F)
    np.save(os.path.join(a.out, "labels.npy"), Y)
    np.save(os.path.join(a.out, "groups.npy"), G)
    if a.face:
        if unknown_labels:
            print(f"  ! метки подсказок не из каталога (пропущены): {', '.join(sorted(unknown_labels))}")
        rare = TONGUE_DIRS + cat["new_channels"]
        print("  редкие каналы, кадров с меткой ≥ 0.5: " + ", ".join(
            f"{c}={int((Y[:, channels.index(c)] >= 0.5).sum())}" for c in rare))
        # Где сток расходится с подсказкой сильнее всего (на твоём лице).
        gaps = []
        for item, chans in item_stats.items():
            for ch, (ssum, n, target) in chans.items():
                if n:
                    gaps.append((abs(ssum / n - target), item, ch, target, ssum / n, n))
        gaps.sort(reverse=True)
        report["stock_vs_prompt"] = [dict(item=i, channel=c, target=t, stock_mean=round(m, 3), frames=n)
                                     for _, i, c, t, m, n in gaps]
        worst = [g for g in gaps if g[0] >= 0.25][:15]
        if worst:
            print("  сток против подсказки (наибольшие расхождения):")
            for gap, item, ch, target, mean, n in worst:
                print(f"    {item:<18} {ch:<16} цель {target:.1f}  сток {mean:.2f}  ({n} кадров)")
    meta = {"height": int(F.shape[1]), "width": int(F.shape[2]), "channels": channels,
            "source": {"kind": "stock-distillation", "labels": a.labels, "crop": a.crop,
                       "size": a.size, "every": a.every, "sessions": a.sessions,
                       "tongue": a.tongue and {"settle": a.settle, "tail": a.tail, "neg_tongue": a.neg_tongue},
                       "face": a.face and {"catalog": os.path.abspath(cat_path), "neg_tongue": a.neg_tongue,
                                           "neg_cheek": a.neg_cheek}}}
    if a.flip:
        meta["flip"] = {"swap": flip_pairs(channels), "mirror": []}
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    std = np.nanstd(Y, axis=0)
    report["channels"] = {n: {"mean": round(float(np.nanmean(Y[:, i])), 4), "std": round(float(std[i]), 4),
                              "known": int((~np.isnan(Y[:, i])).sum())}
                          for i, n in enumerate(channels)}
    quiet = [n for i, n in enumerate(channels) if std[i] < 0.01]
    with open(os.path.join(a.out, "pair-report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if a.tongue:
        total_pos = {d: sum(s["tongue"]["positive"][d] for s in report["sessions"]) for d in TONGUE_DIRS}
        lacking = [d for d, n in total_pos.items() if n == 0]
        if lacking:
            print(f"  ВНИМАНИЕ: нет ни одного кадра с {', '.join(lacking)} — нужны записи с --prompts tongue")
    print(f"датасет → {a.out}: {F.shape[0]} кадров {F.shape[2]}x{F.shape[1]}, {len(channels)} каналов "
          f"({a.labels}), сессий {len(a.sessions)}")
    if quiet:
        print(f"  почти не двигались (std<0.01): {', '.join(quiet)} — эти выражения стоит показать в следующих записях "
              f"(если записи были только по части групп, --groups, это ожидаемо)")


if __name__ == "__main__":
    main()

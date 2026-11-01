#!/usr/bin/env python3
r"""MouthLink — мастер записи и дообучения: одна команда на каждый шаг, без ручных путей.

  python tools/mouthlink.py status            что есть: модель, датасеты, новые записи, что делать дальше
  python tools/mouthlink.py record face       записать сессию лица (или: record tongue;
                                              опции приёмника можно добавить: --groups tongue,cheeks)
  python tools/mouthlink.py update            дообучить на новых записях, проверить и поставить в VRCFT
  python tools/mouthlink.py rollback          вернуть предыдущую модель
  python tools/mouthlink.py restore           вернуть модель и MouthLink.json в модуль (после переустановки)

record: имя следующей записи выбирается само (f09, t22 …). На время записи модуль VRCFT
отпускает камеру ("Source": "udp" в MouthLink.json), после — возвращается как было.
VRCFT при этом должен быть запущен: бридж включает камеры только при подключённом клиенте.

update:
  1. новые записи (ещё не вошедшие ни в один датасет) спариваются в новый датасет;
  2. текущая модель дообучается на старых данных (их доля подбирается сама) + новых,
     последняя новая запись в обучение НЕ входит — она для проверки;
  3. проверка на этой записи: новая модель против текущей — насколько узнаёт язык, щёки,
     губы (узнаёт), сколько ложных срабатываний (ложных), насколько близка к стоку на
     остальном (MAE стока). Новая ставится, только если не хуже текущей;
  4. установка в модуль VRCFT (подхватывается на лету) + копия в резерв.
  Для проверки нужно минимум 2 новые записи (каждая — отдельная посадка шлема).

Опции update: --oversample TongueRight:2 (слабая сторона чаще), --epochs N, --force (поставить
даже если проверка не прошла), --no-check (одна запись, без проверки), --dry-run.

Настройки рабочего места — workspace.json в корне репозитория (в git не попадает): папки
записей и датасетов, модуль VRCFT, IP шлема, история моделей. Создаётся сам при первом запуске.
"""
import argparse, glob, hashlib, json, os, re, shutil, subprocess, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "training")
WS_PATH = os.path.join(ROOT, "workspace.json")
STOCK = {"CheekPuff", "CheekSquintL", "CheekSquintR", "JawForward", "JawLeft", "JawOpen", "JawRight", "MouthClose",
         "MouthDimpleL", "MouthDimpleR", "MouthFrownL", "MouthFrownR", "MouthFunnel", "MouthLeft", "MouthLowerDownL",
         "MouthLowerDownR", "MouthPressL", "MouthPressR", "MouthPucker", "MouthRight", "MouthRollLower",
         "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper", "MouthSmileL", "MouthSmileR", "MouthStretchL",
         "MouthStretchR", "MouthUpperUpL", "MouthUpperUpR", "NoseSneerL", "NoseSneerR", "TongueOut"}


def say(msg=""):
    print(msg, flush=True)


def die(msg):
    say(f"\nОШИБКА: {msg}")
    sys.exit(1)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


# ---------------------------------------------------------------------------------------------
# рабочее место

def find_module_dir():
    base = os.path.join(os.environ.get("APPDATA", ""), "VRCFaceTracking", "CustomLibs")
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.exists(os.path.join(d, "UnifiedPicoModule.dll")):
            return d
    return ""


def detect_workspace():
    rec_dirs = [d for d in (r"G:\mouthrecon", r"D:\mouthrecon", os.path.dirname(ROOT)) if os.path.isdir(d)]
    rec_dirs = list(dict.fromkeys(os.path.normcase(os.path.abspath(d)) for d in rec_dirs))
    data_dirs = [os.path.join(d, "data") for d in rec_dirs if os.path.isdir(os.path.join(d, "data"))] \
        or [os.path.join(rec_dirs[0] if rec_dirs else ROOT, "data")]
    ws = {"recordings": rec_dirs, "data_dirs": data_dirs, "runs_dir": os.path.join(TRAIN, "runs"),
          "module_dir": find_module_dir(), "headset": "", "history": [], "old_budget": 200000,
          "size": 160, "every": 2}
    # Текущая модель: та, что стоит в модуле (по совпадению файла), иначе самая свежая.
    runs = sorted(glob.glob(os.path.join(ws["runs_dir"], "*", "mouth.onnx")), key=os.path.getmtime)
    installed = os.path.join(ws["module_dir"], "MouthLink", "mouth.onnx") if ws["module_dir"] else ""
    cur = None
    if installed and os.path.exists(installed):
        h = md5(installed)
        cur = next((r for r in runs if md5(r) == h), None)
    if cur is None and runs:
        cur = runs[-1]
    if cur:
        ws["history"] = [os.path.basename(os.path.dirname(cur))]
    return ws


def load_ws(create=True):
    if os.path.exists(WS_PATH):
        return json.load(open(WS_PATH, encoding="utf-8"))
    if not create:
        die("нет workspace.json — запусти: python tools/mouthlink.py status")
    ws = detect_workspace()
    save_ws(ws)
    say(f"Создан {WS_PATH} (папки найдены автоматически; при необходимости поправь его вручную).")
    return ws


def save_ws(ws):
    tmp = WS_PATH + ".tmp"
    json.dump(ws, open(tmp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    os.replace(tmp, WS_PATH)


def current_run(ws):
    if not ws.get("history"):
        return None
    run = os.path.join(ws["runs_dir"], ws["history"][-1])
    return run if os.path.exists(os.path.join(run, "best.pt")) else None


def datasets(ws):
    out = []
    for meta in sorted(m for dd in ws["data_dirs"] for m in glob.glob(os.path.join(dd, "*", "meta.json"))):
        d = os.path.dirname(meta)
        if not os.path.exists(os.path.join(d, "frames.npy")):
            continue
        m = json.load(open(meta, encoding="utf-8"))
        sessions = [stem(s) for s in (m.get("source") or {}).get("sessions") or []]
        n = int(np.load(os.path.join(d, "frames.npy"), mmap_mode="r").shape[0])
        out.append({"path": d, "sessions": sessions, "frames": n, "channels": m["channels"],
                    "size": m["height"]})
    return out


def recordings(ws):
    """Все записи с метками стока: имя → путь (первая найденная)."""
    recs = {}
    for d in ws["recordings"]:
        for p in sorted(glob.glob(os.path.join(d, "*.mlrec"))):
            if os.path.exists(os.path.splitext(p)[0] + ".stock.csv"):
                recs.setdefault(stem(p), p)
    return recs


def new_recordings(ws, ds=None):
    used = {s for d in (ds if ds is not None else datasets(ws)) for s in d["sessions"]}
    return {k: v for k, v in recordings(ws).items() if k not in used}


def module_config(ws):
    p = os.path.join(ws.get("module_dir") or "", "MouthLink.json")
    return p if ws.get("module_dir") and os.path.exists(p) else ""


def read_lo_hi(ws):
    p = module_config(ws)
    lo, hi = 0.15, 0.55
    if p:
        text = open(p, encoding="utf-8").read()
        m = re.search(r'"Lo"\s*:\s*([0-9.]+)', text)
        lo = float(m.group(1)) if m else lo
        m = re.search(r'"Hi"\s*:\s*([0-9.]+)', text)
        hi = float(m.group(1)) if m else hi
    return lo, hi


def backup_dir(ws):
    return ws.get("backup_dir") or os.path.join(os.path.dirname(ROOT), "vrcft-module-backup")


# ---------------------------------------------------------------------------------------------
# status

def cmd_status(a):
    ws = load_ws()
    ds = datasets(ws)
    run = current_run(ws)
    say(f"Записи:          {', '.join(ws['recordings']) or '(не найдены)'}")
    say(f"Датасеты:        {', '.join(ws['data_dirs'])}")
    for d in ds:
        say(f"  {os.path.basename(d['path']):<16} {d['frames']:>8} кадров, сессии: {', '.join(d['sessions'])}")
    say(f"Модуль VRCFT:    {ws.get('module_dir') or '(не найден — VRCFT с UnifiedPicoModule-MouthLink не установлен?)'}")
    say(f"Текущая модель:  {os.path.basename(run) if run else '(нет)'}"
        + (f"   (история: {' → '.join(ws['history'][-5:])})" if len(ws.get('history', [])) > 1 else ""))
    installed = os.path.join(ws.get("module_dir") or "", "MouthLink", "mouth.onnx")
    if run and ws.get("module_dir"):
        if not os.path.exists(installed):
            say("  ! в модуле нет модели — python tools/mouthlink.py restore")
        elif os.path.exists(os.path.join(run, "mouth.onnx")) and md5(installed) != md5(os.path.join(run, "mouth.onnx")):
            say("  ! в модуле стоит другая модель, не текущая — restore поставит текущую")
    new = new_recordings(ws, ds)
    say(f"Новые записи:    {', '.join(new) if new else 'нет'}")
    say("")
    if not new:
        say("Дальше: python tools/mouthlink.py record face   (или record tongue) — записать новую сессию.")
    elif len(new) == 1:
        say("Дальше: запиши ещё одну сессию (сними и надень шлем) — для проверки нужно минимум две,")
        say("        или python tools/mouthlink.py update --no-check (без проверки).")
    else:
        say("Дальше: python tools/mouthlink.py update")


# ---------------------------------------------------------------------------------------------
# record

def set_source(path, value):
    """Меняет "Source" в MouthLink.json модуля, возвращает прежнее значение (или None)."""
    text = open(path, encoding="utf-8").read()
    m = re.search(r'("Source"\s*:\s*")([^"]*)(")', text)
    if not m:
        return None
    old = m.group(2)
    if old != value:
        tmp = path + ".tmp"
        open(tmp, "w", encoding="utf-8").write(text[:m.start(2)] + value + text[m.end(2):])
        os.replace(tmp, path)
    return old


def next_name(ws, prefix):
    stems = {stem(p) for d in ws["recordings"] for p in glob.glob(os.path.join(d, "*.mlrec"))}
    nums = [int(m.group(1)) for k in stems if (m := re.fullmatch(re.escape(prefix) + r"(\d+)", k))]
    return f"{prefix}{(max(nums) + 1) if nums else 1:02d}"


def cmd_record(a):
    ws = load_ws()
    if not ws["recordings"]:
        die("не найдена папка для записей — укажи \"recordings\" в workspace.json")
    host = a.host or ws.get("headset")
    if not host:
        host = input("IP шлема (см. Wi-Fi в настройках шлема или в бридже): ").strip()
        if not host:
            die("нужен IP шлема")
    if host != ws.get("headset"):
        ws["headset"] = host
        save_ws(ws)
    name = a.name or next_name(ws, "f" if a.prompts == "face" else "t")
    out = os.path.join(ws["recordings"][0], name + ".mlrec")
    if os.path.exists(out):
        die(f"{out} уже есть")
    cfg = module_config(ws)
    old = None
    if cfg:
        old = set_source(cfg, "udp")
        if old is None:
            say("! в MouthLink.json модуля нет \"Source\" — если камера занята, поставь \"Source\": \"udp\" вручную")
        elif old != "udp":
            say("Модуль VRCFT отпускает камеру на время записи (\"Source\": \"udp\")…")
            time.sleep(2.5)
    cmd = ["dotnet", "run", "--project", os.path.join(ROOT, "MouthLink.Receiver"), "--",
           "--host", host, "--fps", "0", "--record", out, "--prompts", a.prompts] + a.extra
    say(f"Запись {name} → {out}")
    say("Шлем на голове, VRCFT запущен. Ctrl+C — остановить раньше.\n")
    try:
        subprocess.run(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        pass
    finally:
        if cfg and old not in (None, "udp"):
            set_source(cfg, old)
            say(f"\nМодуль VRCFT снова берёт камеру (\"Source\": \"{old}\").")
    if os.path.exists(out):
        n = len(new_recordings(ws))
        say(f"\nГотово: {name}. Новых записей: {n}. "
            + ("Сними и надень шлем и запиши ещё одну — или: python tools/mouthlink.py update" if n < 2
               else "Можно дообучать: python tools/mouthlink.py update"))


# ---------------------------------------------------------------------------------------------
# проверка: новая модель против текущей на отложенной записи

def evaluate(onnx_paths, ds_path, session, lo, hi):
    import onnxruntime as ort
    meta = json.load(open(os.path.join(ds_path, "meta.json"), encoding="utf-8"))
    names = [stem(s) for s in meta["source"]["sessions"]]
    gid = names.index(session)
    groups = np.load(os.path.join(ds_path, "groups.npy"))
    idx = np.flatnonzero(groups == gid)
    X = np.load(os.path.join(ds_path, "frames.npy"), mmap_mode="r")
    Y = np.load(os.path.join(ds_path, "labels.npy"))[idx]
    ch = meta["channels"]
    new = np.array([c not in STOCK for c in ch])
    thr = np.where(new, lo + 0.5 * (hi - lo), 0.5)      # как модуль показывает: новые после Lo/Hi
    results = []
    for path in onnx_paths:
        side = json.load(open(os.path.splitext(path)[0] + ".json", encoding="utf-8"))
        if side["channels"] != ch:
            die(f"{path}: каналы модели не совпадают с датасетом")
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        out = np.empty((len(idx), len(ch)), np.float32)
        for b in range(0, len(idx), 256):
            x = (X[idx[b:b + 256]].astype(np.float32) / 255.0)[:, None]
            out[b:b + 256] = s.run(None, {name: x})[0]
        out = np.clip(out, 0, 1)
        per = {}
        for i, c in enumerate(ch):
            if not new[i]:
                continue
            y = Y[:, i]
            pos, neg = y >= 0.5, y == 0
            per[c] = (float((out[pos, i] >= thr[i]).mean()) if pos.sum() >= 10 else None,
                      float((out[neg, i] >= thr[i]).mean()) if neg.sum() >= 10 else None, int(pos.sum()))
        rec = [v[0] for v in per.values() if v[0] is not None]
        fal = [v[1] for v in per.values() if v[1] is not None]
        st = ~new[None, :] & np.isfinite(Y)
        mae = float(np.abs(out - np.nan_to_num(Y))[st].mean()) if st.any() else 0.0
        r, f = (float(np.mean(rec)) if rec else 0.0), (float(np.mean(fal)) if fal else 0.0)
        results.append({"recall": r, "false": f, "stock_mae": mae, "score": r - 2 * f, "per": per})
    return results, len(idx)


def verdict(cur, cand):
    reasons = []
    if cand["score"] < cur["score"] - 0.01:
        reasons.append(f"общий счёт хуже ({cand['score']:.3f} < {cur['score']:.3f})")
    if cand["false"] > cur["false"] + 0.02:
        reasons.append(f"больше ложных ({cand['false']:.1%} > {cur['false']:.1%})")
    if cand["stock_mae"] > cur["stock_mae"] + 0.01:
        reasons.append(f"остальное лицо хуже (MAE стока {cand['stock_mae']:.3f} > {cur['stock_mae']:.3f})")
    return reasons


def print_compare(cur, cand, session, n):
    def p(v):
        return "   —  " if v is None else f"{v:6.0%}"
    say(f"\nПроверка на записи {session} ({n} кадров; модели её не видели):")
    say(f"  {'':<26}{'текущая':>10}{'новая':>10}")
    say(f"  {'узнаёт (в среднем)':<26}{cur['recall']:>10.1%}{cand['recall']:>10.1%}")
    say(f"  {'ложных (в среднем)':<26}{cur['false']:>10.1%}{cand['false']:>10.1%}")
    say(f"  {'MAE стока (остальное лицо)':<26}{cur['stock_mae']:>10.3f}{cand['stock_mae']:>10.3f}")
    say(f"  {'счёт: узнаёт − 2×ложных':<26}{cur['score']:>10.3f}{cand['score']:>10.3f}")
    say(f"  по каналам (узнаёт / ложных):")
    for c in cur["per"]:
        a, b = cur["per"][c], cand["per"][c]
        if a[0] is None and a[1] is None:
            continue
        say(f"    {c:<20} {p(a[0])} / {p(a[1])}   →   {p(b[0])} / {p(b[1])}")


# ---------------------------------------------------------------------------------------------
# update

def run(cmd, cwd=ROOT):
    say("\n> " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        die(f"шаг завершился с ошибкой (код {r.returncode})")


def next_run_name(ws, base):
    nums = [int(m.group(1)) for p in glob.glob(os.path.join(ws["runs_dir"], "*"))
            if (m := re.search(r"-ft(\d+)$", os.path.basename(p)))]
    prefix = re.sub(r"-ft\d+$", "", base) if base else "face"
    return f"{prefix}-ft{(max(nums) + 1) if nums else 1}"


def install(ws, run_dir):
    mod = ws.get("module_dir")
    if not mod or not os.path.isdir(mod):
        say("! модуль VRCFT не найден — модель не установлена (укажи \"module_dir\" в workspace.json)")
        return False
    dst = os.path.join(mod, "MouthLink")
    os.makedirs(dst, exist_ok=True)
    # Сначала описание каналов, затем модель: модуль перечитывает модель по изменению mouth.onnx.
    for name in ("mouth.json", "mouth.onnx"):
        tmp = os.path.join(dst, name + ".tmp")
        shutil.copyfile(os.path.join(run_dir, name), tmp)
        os.replace(tmp, os.path.join(dst, name))
    save_backup(ws)
    return True


def save_backup(ws):
    """Резерв модели и MouthLink.json вне папки модуля: VRCFT стирает её при переустановке."""
    mod = ws.get("module_dir")
    if not mod:
        return
    b = backup_dir(ws)
    os.makedirs(os.path.join(b, "MouthLink"), exist_ok=True)
    for rel in ("MouthLink.json", os.path.join("MouthLink", "mouth.json"), os.path.join("MouthLink", "mouth.onnx")):
        src = os.path.join(mod, rel)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(b, rel))


def cmd_update(a):
    ws = load_ws()
    base = current_run(ws)
    if not base:
        die("нет текущей модели (best.pt) — сначала обучи модель с нуля (training/README.md)")
    ds = datasets(ws)
    new = new_recordings(ws, ds)
    if a.recordings:
        recs = {}
        for p in a.recordings:
            recs.update({stem(x): x for x in glob.glob(p)} if any(c in p for c in "*?[") else {stem(p): p})
        new = recs
    if not new:
        die("новых записей нет — сначала: python tools/mouthlink.py record face")
    check = not a.no_check
    if check and len(new) < 2:
        die("для проверки нужно минимум 2 новые записи (разные посадки шлема): запиши ещё одну "
            "или запусти с --no-check")
    names = sorted(new, key=lambda k: os.path.getmtime(new[k]))
    new = {k: new[k] for k in names}
    holdout = names[-1] if check else None      # самая свежая запись — для проверки
    base_meta = json.load(open(os.path.join(base, "mouth.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(base, "mouth.json")) else None
    size = base_meta["height"] if base_meta else ws.get("size", 160)
    channels = base_meta["channels"] if base_meta else None
    old = [d for d in ds if d["size"] == size and (channels is None or d["channels"] == channels)]
    skipped = [os.path.basename(d["path"]) for d in ds if d not in old]
    total_old = sum(d["frames"] for d in old)
    frac = min(1.0, ws.get("old_budget", 200000) / total_old) if total_old else 1.0
    run_name = a.name or next_run_name(ws, os.path.basename(base))
    out_run = os.path.join(ws["runs_dir"], run_name)
    new_ds = os.path.join(ws["data_dirs"][0], "upd-" + time.strftime("%Y%m%d-%H%M"))

    say(f"Текущая модель: {os.path.basename(base)}  →  новая: {run_name}")
    say(f"Новые записи: {', '.join(names)}" + (f"  (проверка на {holdout}, в обучение не идёт)" if holdout else ""))
    say(f"Старые данные: {len(old)} датасетов, {total_old} кадров, берётся {frac:.0%}"
        + (f"; пропущены (другие каналы/размер): {', '.join(skipped)}" if skipped else ""))
    if a.dry_run:
        return

    # 1. спаривание новых записей
    say("\n=== Шаг 1/4: новые записи → датасет ===")
    run([sys.executable, os.path.join(ROOT, "tools", "mlrec_pair.py"), new_ds, *new.values(),
         "--face", "--size", str(size), "--every", str(ws.get("every", 2))])

    # 2. дообучение
    say("\n=== Шаг 2/4: дообучение ===")
    data = [f"{d['path']}@{frac:.3g}" if frac < 1 else d["path"] for d in old] + [new_ds]
    cmd = [sys.executable, "train.py", "--data", ",".join(data), "--init", os.path.join(base, "best.pt"),
           "--out", out_run]
    cmd += ["--val-groups", holdout] if holdout else ["--val-frac", "0"]
    if a.epochs:
        cmd += ["--epochs", str(a.epochs)]
    if a.oversample:
        cmd += ["--oversample", a.oversample]
    run(cmd, cwd=TRAIN)
    run([sys.executable, "export_onnx.py", "--ckpt", os.path.join(out_run, "best.pt"),
         "--out", os.path.join(out_run, "mouth.onnx")], cwd=TRAIN)

    # 3. проверка
    if holdout:
        say("\n=== Шаг 3/4: проверка новой модели против текущей ===")
        base_onnx = os.path.join(base, "mouth.onnx")
        if not os.path.exists(base_onnx):
            run([sys.executable, "export_onnx.py", "--ckpt", os.path.join(base, "best.pt"), "--out", base_onnx],
                cwd=TRAIN)
        lo, hi = read_lo_hi(ws)
        (cur, cand), n = evaluate([base_onnx, os.path.join(out_run, "mouth.onnx")], new_ds, holdout, lo, hi)
        print_compare(cur, cand, holdout, n)
        reasons = verdict(cur, cand)
        json.dump({"holdout": holdout, "current": os.path.basename(base), "candidate": run_name,
                   "current_metrics": cur, "candidate_metrics": cand, "rejected_because": reasons},
                  open(os.path.join(out_run, "check.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        if reasons:
            say("\nНовая модель НЕ лучше текущей: " + "; ".join(reasons) + ".")
            if not a.force:
                say(f"Оставляю {os.path.basename(base)}. Записи сохранены в датасете {os.path.basename(new_ds)} "
                    "и войдут в следующее дообучение.")
                say("Варианты: записать ещё сессии с нужными позами; --oversample КАНАЛ:2 для слабого канала; "
                    "--force — поставить всё равно.")
                return
            say("--force: ставлю всё равно.")
        else:
            say("\nПроверка пройдена: новая модель не хуже текущей.")

    # 4. установка
    say("\n=== Шаг 4/4: установка в VRCFT ===")
    ws["history"] = (ws.get("history") or []) + [run_name]
    save_ws(ws)
    if install(ws, out_run):
        say(f"\nУстановлено: {run_name} → {os.path.join(ws['module_dir'], 'MouthLink')} "
            "(VRCFT подхватит за секунду; резерв — в " + backup_dir(ws) + ")")
    say("Откатить: python tools/mouthlink.py rollback")


def cmd_rollback(a):
    ws = load_ws(create=False)
    hist = ws.get("history") or []
    if len(hist) < 2:
        die("откатывать некуда: в истории одна модель")
    prev = os.path.join(ws["runs_dir"], hist[-2])
    if not os.path.exists(os.path.join(prev, "mouth.onnx")):
        die(f"нет {prev}\\mouth.onnx")
    ws["history"] = hist[:-1]
    save_ws(ws)
    install(ws, prev)
    say(f"Возвращена модель {hist[-2]} (была {hist[-1]}).")


def cmd_restore(a):
    ws = load_ws(create=False)
    mod = ws.get("module_dir")
    if not mod or not os.path.isdir(mod):
        die("модуль VRCFT не найден: поставь UnifiedPicoModule-MouthLink и запусти снова")
    b = backup_dir(ws)
    cfg_b = os.path.join(b, "MouthLink.json")
    if os.path.exists(cfg_b):
        shutil.copyfile(cfg_b, os.path.join(mod, "MouthLink.json"))
        say(f"MouthLink.json восстановлен из {b}")
    run_dir = current_run(ws)
    if run_dir and os.path.exists(os.path.join(run_dir, "mouth.onnx")):
        install(ws, run_dir)
        say(f"Модель {os.path.basename(run_dir)} установлена.")
    elif os.path.exists(os.path.join(b, "MouthLink", "mouth.onnx")):
        os.makedirs(os.path.join(mod, "MouthLink"), exist_ok=True)
        for n in ("mouth.json", "mouth.onnx"):
            shutil.copyfile(os.path.join(b, "MouthLink", n), os.path.join(mod, "MouthLink", n))
        say("Модель восстановлена из резерва.")
    else:
        die("нет ни текущей модели, ни резерва")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    r = sub.add_parser("record")
    r.add_argument("prompts", choices=["face", "tongue"], nargs="?", default="face")
    r.add_argument("--host", help="IP шлема (запоминается)")
    r.add_argument("--name", help="имя записи (по умолчанию следующее: f09, t22 …)")
    r.set_defaults(fn=cmd_record)
    u = sub.add_parser("update")
    u.add_argument("recordings", nargs="*", help="записи (по умолчанию все новые)")
    u.add_argument("--oversample", default="")
    u.add_argument("--epochs", type=int)
    u.add_argument("--name", help="имя прогона (по умолчанию face-ftN)")
    u.add_argument("--force", action="store_true", help="установить, даже если проверка не прошла")
    u.add_argument("--no-check", action="store_true", help="без проверки (например, одна новая запись)")
    u.add_argument("--dry-run", action="store_true", help="только показать план")
    u.set_defaults(fn=cmd_update)
    sub.add_parser("rollback").set_defaults(fn=cmd_rollback)
    sub.add_parser("restore").set_defaults(fn=cmd_restore)
    a, extra = ap.parse_known_args()
    if extra and a.cmd != "record":
        ap.error("неизвестные опции: " + " ".join(extra))
    a.extra = [x for x in extra if x != "--"]     # record: прочие опции идут в MouthLink.Receiver
    a.fn(a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Обучение модели рта на датасете формата training/README.md.

Главное:
  * Деление train/val **по сессиям** (groups.npy): соседние кадры одной записи почти
    одинаковы, и случайное деление по кадрам даёт слишком оптимистичную валидацию.
    --cv: каждая сессия по очереди исключается целиком (честная оценка на новой посадке).
  * Сильные аугментации на батче: масштаб, поворот, сдвиг (разная посадка шлема),
    гамма/яркость/шум, отражение по горизонтали с перестановкой меток по правилам
    meta.json["flip"] (левые↔правые каналы, позиционные каналы v→1-v).
  * Маскированный лосс: NaN в labels = канал неизвестен, в лосс не идёт.
  * Предобученный MobileNetV3-small (--arch mnv3 --pretrained): BN энкодера заморожен,
    lr энкодера в ~30 раз меньше, чем у головы.
  * Дообучение (--init runs\face-final\best.pt): старт с уже обученной модели, по умолчанию
    3 эпохи и lr 5e-4. Несколько датасетов сразу: --data data\face,data\face2 — новые записи
    достаточно спарить отдельно, старые не пересобирать.
  * Если датасет не помещается в память видеокарты, кадры остаются в ОЗУ и идут на GPU батчами.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from model import build_model, freeze_encoder_bn

AUG = {
    # rot: ± градусы; scale: [min,max] масштаба сетки выборки; shift: ± в долях
    # нормированной координаты [-1,1]; flip: отражение; photometric: гамма/яркость/шум.
    "none":   dict(rot=0,  scale=(1.0, 1.0),   shift=0.0,  flip=False, gamma=(1.0, 1.0), gain=(1.0, 1.0),   offset=0.0,  noise=0.0),
    "light":  dict(rot=7,  scale=(1.0, 1.0),   shift=0.12, flip=False, gamma=(1.0, 1.0), gain=(0.85, 1.15), offset=0.05, noise=0.016),
    "strong": dict(rot=10, scale=(0.75, 1.35), shift=0.18, flip=True,  gamma=(0.7, 1.4), gain=(0.8, 1.2),   offset=0.05, noise=0.02),
}


def load_dataset(path, mmap=False):
    frames = np.load(os.path.join(path, "frames.npy"), mmap_mode="r" if mmap else None)
    labels = np.load(os.path.join(path, "labels.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(path, "meta.json"), encoding="utf-8"))
    gpath = os.path.join(path, "groups.npy")
    if os.path.exists(gpath):
        raw = np.load(gpath, allow_pickle=False)
        uniq, groups = np.unique(raw, return_inverse=True)   # любые id → 0..G-1
        # Имена сессий — по файлам записей из meta.json (f01, s03, …), иначе номер.
        files = (meta.get("source") or {}).get("sessions") or []
        names = [os.path.splitext(os.path.basename(files[v]))[0] if 0 <= v < len(files) else str(v)
                 for v in uniq.tolist()]
    else:
        print("ВНИМАНИЕ: нет groups.npy — весь датасет считается одной сессией")
        groups = np.zeros(len(frames), np.int64)
        names = ["0"]
    assert len(frames) == len(labels) == len(groups)
    return frames, labels, groups.astype(np.int64), names, meta


def load_datasets(paths):
    """Один или несколько датасетов (одинаковые каналы и размер кадра) → один."""
    if len(paths) == 1:
        return load_dataset(paths[0])
    parts = [load_dataset(p, mmap=True) for p in paths]
    meta0 = parts[0][4]
    for p, part in zip(paths[1:], parts[1:]):
        m = part[4]
        if m["channels"] != meta0["channels"] or (m["height"], m["width"]) != (meta0["height"], meta0["width"]):
            raise SystemExit(f"{p}: каналы или размер кадра не совпадают с {paths[0]} "
                             f"(собирай mlrec_pair с теми же --face/--tongue и --size)")
    n = sum(len(part[0]) for part in parts)
    frames = np.empty((n, meta0["height"], meta0["width"]), np.uint8)
    labels, groups, names, o, g0 = [], [], [], 0, 0
    for p, (f, y, g, nm, _) in zip(paths, parts):
        frames[o:o + len(f)] = f
        o += len(f)
        labels.append(y); groups.append(g + g0); g0 += len(nm)
        names += nm
    dup = {x for x in names if names.count(x) > 1}
    if dup:
        print(f"ВНИМАНИЕ: сессии {sorted(dup)} есть в нескольких датасетах — кадры повторятся")
        seen = {}
        for i, x in enumerate(names):
            seen[x] = seen.get(x, 0) + 1
            if seen[x] > 1:
                names[i] = f"{x}#{seen[x]}"
    meta = dict(meta0)
    meta["source"] = {"datasets": paths}
    return frames, np.concatenate(labels), np.concatenate(groups), names, meta


def frames_batch(X, ii, dev):
    """Кадры батча на устройстве модели в [0,1]; X может лежать в ОЗУ, а не на GPU."""
    if torch.is_tensor(ii) and ii.device != X.device:
        ii = ii.cpu()
    x = X[ii]
    if x.device != dev:
        x = x.to(dev, non_blocking=True)
    return x.float()[:, None] / 255.0


def flip_maps(meta, K):
    """Перестановка каналов и маска «зеркальных» каналов для отражения кадра."""
    chans = meta["channels"]
    spec = meta.get("flip")
    if spec is None:
        return None, None
    perm = list(range(K))
    for a, b in spec.get("swap", []):
        ia, ib = chans.index(a), chans.index(b)
        perm[ia], perm[ib] = ib, ia
    mirror = torch.zeros(K, dtype=torch.bool)
    for c in spec.get("mirror", []):
        mirror[chans.index(c)] = True
    return torch.tensor(perm), mirror


def augment(x, y, cfg, perm, mirror, gen):
    """x: [B,1,H,W] float [0,1]; y: [B,K] (NaN допустим). Возвращает аугментированные копии."""
    B, dev = x.shape[0], x.device
    r = lambda *s: torch.rand(*s, device=dev, generator=gen)
    ang = (r(B) - 0.5) * 2 * math.radians(cfg["rot"])
    smin, smax = cfg["scale"]
    sc = smin + r(B) * (smax - smin)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = ang.cos() * sc
    theta[:, 0, 1] = -ang.sin() * sc
    theta[:, 1, 0] = ang.sin() * sc
    theta[:, 1, 1] = ang.cos() * sc
    theta[:, :, 2] = (r(B, 2) - 0.5) * 2 * cfg["shift"]
    y = y.clone()
    if cfg["flip"] and perm is not None:
        f = r(B) < 0.5
        theta[f, 0, :2] *= -1                                   # зеркало по горизонтали
        yf = y[f][:, perm.to(dev)]
        m = mirror.to(dev)
        yf[:, m] = 1.0 - yf[:, m]
        y[f] = yf
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    g0, g1 = cfg["gamma"]
    x = x.clamp_min(1e-6) ** (g0 + r(B, 1, 1, 1) * (g1 - g0))
    a0, a1 = cfg["gain"]
    x = x * (a0 + r(B, 1, 1, 1) * (a1 - a0)) + (r(B, 1, 1, 1) - 0.5) * 2 * cfg["offset"]
    if cfg["noise"] > 0:
        x = x + torch.randn(x.shape, device=dev, generator=gen) * (r(B, 1, 1, 1) * cfg["noise"])
    return x.clamp(0, 1), y


def masked_loss(pred, target):
    mask = ~torch.isnan(target)
    n = mask.sum()
    if n == 0:
        return pred.sum() * 0.0
    diff = (pred - torch.nan_to_num(target)) * mask
    return (diff * diff).sum() / n


@torch.no_grad()
def evaluate(net, X, Y, G, idx, batch, n_groups):
    """MAE по каналам и по сессиям (только известные значения)."""
    net.eval()
    K = Y.shape[1]
    abs_sum = torch.zeros(K, dtype=torch.float64)
    cnt = torch.zeros(K, dtype=torch.float64)
    g_abs = torch.zeros(n_groups, dtype=torch.float64)
    g_cnt = torch.zeros(n_groups, dtype=torch.float64)
    for s in range(0, len(idx), batch):
        ii = idx[s:s + batch]
        p = net(frames_batch(X, ii, next(net.parameters()).device)).cpu().double()
        t = Y[ii].cpu().double()
        m = ~torch.isnan(t)
        e = (p - torch.nan_to_num(t)).abs() * m
        abs_sum += e.sum(0)
        cnt += m.sum(0)
        gi = G[ii].cpu()
        g_abs.index_add_(0, gi, e.sum(1))
        g_cnt.index_add_(0, gi, m.sum(1).double())
    per_ch = (abs_sum / cnt.clamp_min(1)).tolist()
    overall = float(abs_sum.sum() / cnt.sum().clamp_min(1))
    per_g = {int(g): float(g_abs[g] / g_cnt[g]) for g in range(n_groups) if g_cnt[g] > 0}
    return overall, per_ch, per_g


# Редкие каналы (размечены только подсказками, почти все метки — 0): направления языка и
# каналы сценария лица, которых нет у стока (prompts/face.json → new_channels).
DIRECTION_CHANNELS = ("TongueLeft", "TongueRight", "TongueUp", "TongueDown",
                      "CheekPuffLeft", "CheekPuffRight", "CheekSuckLeft", "CheekSuckRight",
                      "MouthTightenerLeft", "MouthTightenerRight", "TongueRoll")


@torch.no_grad()
def direction_scores(net, X, Y, idx, batch, channels):
    """Для редких каналов-направлений средний MAE ничего не говорит (почти все метки — 0).
    Считаем: «узнаёт» — доля кадров с меткой 1, где предсказание ≥ 0.5; «ложных» — доля
    кадров с меткой 0, где предсказание ≥ 0.5."""
    cols = [i for i, c in enumerate(channels) if c in DIRECTION_CHANNELS]
    if not cols:
        return {}
    net.eval()
    hit = {i: [0, 0] for i in cols}
    false = {i: [0, 0] for i in cols}
    for s in range(0, len(idx), batch):
        ii = idx[s:s + batch]
        p = net(frames_batch(X, ii, next(net.parameters()).device)).cpu()
        t = Y[ii].cpu()
        for i in cols:
            ti, pi = t[:, i], p[:, i]
            pos, neg = ti >= 0.5, ti < 0.5   # NaN не попадает ни туда, ни туда
            hit[i][0] += int((pi[pos] >= 0.5).sum()); hit[i][1] += int(pos.sum())
            false[i][0] += int((pi[neg] >= 0.5).sum()); false[i][1] += int(neg.sum())
    out = {}
    for i in cols:
        out[channels[i]] = dict(recall=hit[i][0] / hit[i][1] if hit[i][1] else None, positives=hit[i][1],
                                false_rate=false[i][0] / false[i][1] if false[i][1] else None)
    return out


def format_directions(d):
    parts = []
    for c, v in d.items():
        if v["recall"] is None:
            parts.append(f"{c}: нет примеров")
        else:
            parts.append(f"{c}: узнаёт {100 * v['recall']:.0f}% из {v['positives']}, ложных {100 * (v['false_rate'] or 0):.1f}%")
    return "  редкие каналы — " + "; ".join(parts)


def train_once(a, X, Y, G, tr_idx, va_idx, K, perm, mirror, n_groups, dev, log=print):
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    net = build_model(a.arch, K, width=a.width, pretrained=a.pretrained and a.init_state is None).to(dev)
    if a.init_state is not None:
        net.load_state_dict(a.init_state)
    enc_lr = a.lr * (a.enc_lr_mult if a.pretrained else 1.0)
    opt = torch.optim.AdamW([
        dict(params=net.encoder.parameters(), lr=enc_lr),
        dict(params=net.head.parameters(), lr=a.lr),
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    cfg = AUG[a.aug]
    best, best_state, hist = float("inf"), None, []
    tr_idx_t = torch.tensor(tr_idx, device=dev)
    # Батч не больше обучающей выборки: на калибровочных сессиях кадров бывает меньше,
    # чем размер батча, и иначе не было бы ни одного шага.
    bs = max(2, min(a.batch, len(tr_idx)))
    steps = a.epochs * (len(tr_idx) // bs)
    if steps < 200:
        log(f"  ВНИМАНИЕ: всего {steps} шагов оптимизатора — модель, скорее всего, недообучится; "
            f"увеличь --epochs (на маленьких данных считай шаги, а не эпохи)")
    for ep in range(1, a.epochs + 1):
        net.train()
        if a.pretrained:
            freeze_encoder_bn(net)
        order = tr_idx_t[torch.randperm(len(tr_idx_t), device=dev, generator=gen)]
        tl, tn = 0.0, 0
        for s in range(0, len(order) - bs + 1, bs):
            ii = order[s:s + bs]
            x = frames_batch(X, ii, dev)
            y = Y[ii]
            if a.aug != "none":
                x, y = augment(x, y, cfg, perm, mirror, gen)
            loss = masked_loss(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(ii); tn += len(ii)
        sched.step()
        row = dict(epoch=ep, train_mse=tl / max(1, tn))
        if len(va_idx):
            vm, vch, vg = evaluate(net, X, Y, G, va_idx, a.batch * 2, n_groups)
            row.update(val_mae=vm)
            if vm < best:
                best = vm
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            log(f"  эпоха {ep:2d}/{a.epochs}  train MSE={row['train_mse']:.4f}  val MAE={vm:.4f}")
        else:
            log(f"  эпоха {ep:2d}/{a.epochs}  train MSE={row['train_mse']:.4f}")
        hist.append(row)
    if best_state is None:   # без валидации — берём последнюю эпоху
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, best_state, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="датасет или несколько через запятую")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=["tiny", "mnv3"], default="tiny")
    ap.add_argument("--pretrained", action="store_true", help="веса ImageNet для mnv3 (скачиваются torchvision)")
    ap.add_argument("--aug", choices=list(AUG), default="strong")
    ap.add_argument("--split", choices=["sessions", "frames"], default="sessions")
    ap.add_argument("--val-frac", type=float, default=0.2, help="доля сессий (или кадров) в валидации; 0 = учить на всём")
    ap.add_argument("--val-groups", default="", help="явный список сессий для валидации, через запятую")
    ap.add_argument("--cv", action="store_true", help="исключать каждую сессию по очереди и вывести сводку")
    ap.add_argument("--init", help="дообучить: стартовать с этого чекпойнта (best.pt); arch берётся из него")
    ap.add_argument("--epochs", type=int, default=None, help="по умолчанию 10, с --init 3")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None, help="по умолчанию 2e-3, с --init 5e-4")
    ap.add_argument("--enc-lr-mult", type=float, default=0.03, help="множитель lr энкодера при --pretrained")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames-on", choices=["auto", "gpu", "cpu"], default="auto",
                    help="где держать кадры: auto — в видеокарте, если влезают с запасом --vram-reserve")
    ap.add_argument("--vram-reserve", type=int, default=2, help="ГБ видеопамяти оставить под модель (auto)")
    a = ap.parse_args()

    a.init_state = None
    init_ck = None
    if a.init:
        init_ck = torch.load(a.init, map_location="cpu", weights_only=False)
        a.arch, a.width = init_ck.get("arch", "tiny"), init_ck.get("width_mult", 16)
        # mnv3 обучался от ImageNet с замороженным BN — дообучаем так же (без скачивания весов).
        a.pretrained = init_ck.get("pretrained", a.arch == "mnv3")
        a.init_state = init_ck["state"]
        if a.cv:
            raise SystemExit("--cv с --init нечестен: стартовая модель уже видела все старые сессии. "
                             "Для оценки новой посадки: --val-groups <новая запись>, или tools/mlrec_eval.py")
    a.epochs = a.epochs or (3 if a.init else 10)
    a.lr = a.lr or (5e-4 if a.init else 2e-3)
    if a.pretrained and a.arch != "mnv3":
        raise SystemExit("--pretrained имеет смысл только с --arch mnv3")

    frames, labels, groups, gnames, meta = load_datasets([p.strip() for p in a.data.split(",") if p.strip()])
    if init_ck is not None:
        if init_ck["channels"] != meta["channels"] or (init_ck["height"], init_ck["width"]) != (meta["height"], meta["width"]):
            raise SystemExit(f"{a.init}: каналы или размер кадра модели не совпадают с датасетом — "
                             f"дообучать можно только на датасете того же вида (--face, --size)")
        print(f"дообучение с {a.init}: эпох {a.epochs}, lr {a.lr}")
    K, n_groups = labels.shape[1], len(gnames)
    perm, mirror = flip_maps(meta, K)
    if AUG[a.aug]["flip"] and perm is None:
        print("ВНИМАНИЕ: в meta.json нет правил 'flip' — отражение отключено")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"датасет: {len(frames)} кадров {meta['width']}x{meta['height']}, каналов={K} {meta['channels']}, "
          f"сессий={n_groups}; arch={a.arch}{'+imagenet' if a.pretrained else ''}, aug={a.aug}, устройство={dev}")

    N = len(frames)
    X = torch.from_numpy(frames)                               # uint8 [N,H,W]
    if dev.type == "cuda":
        # На Windows драйвер при нехватке видеопамяти не падает, а молча выносит часть в общую
        # память (медленно), поэтому решаем заранее: оставляем запас под модель и активации.
        free, _ = torch.cuda.mem_get_info()
        need = X.numel() + (a.vram_reserve << 30)
        if a.frames_on == "cpu" or (a.frames_on == "auto" and need > free):
            print(f"кадры ({X.numel() / 2**30:.1f} ГБ) остаются в ОЗУ (свободно в видеокарте "
                  f"{free / 2**30:.1f} ГБ, запас {a.vram_reserve} ГБ) — на GPU идут батчами")
        else:
            try:
                X = X.to(dev)
                del frames                                     # копия в ОЗУ больше не нужна
                frames = None
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"кадры ({X.numel() / 2**30:.1f} ГБ) не помещаются в видеокарту — остаются в ОЗУ, "
                      f"на GPU идут батчами (медленнее)")
    Y = torch.from_numpy(labels).to(dev)
    G = torch.from_numpy(groups).to(dev)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    if a.cv:
        if n_groups < 2:
            raise SystemExit("--cv нужно минимум 2 сессии")
        folds = []
        for g in range(n_groups):
            va = np.flatnonzero(groups == g)
            tr = np.flatnonzero(groups != g)
            print(f"фолд {g + 1}/{n_groups}: исключена сессия '{gnames[g]}' ({len(va)} кадров)")
            net, _, _ = train_once(a, X, Y, G, tr, va, K, perm, mirror, n_groups, dev, log=lambda *_: None)
            vm, vch, _ = evaluate(net, X, Y, G, va, a.batch * 2, n_groups)
            print(f"  MAE={vm:.4f}  по каналам: " + ", ".join(f"{c}={v:.3f}" for c, v in zip(meta["channels"], vch)))
            dirs = direction_scores(net, X, Y, va, a.batch * 2, meta["channels"])
            if dirs:
                print(format_directions(dirs))
            folds.append(dict(session=gnames[g], mae=vm, per_channel=dict(zip(meta["channels"], vch)),
                              directions=dirs))
        maes = [f["mae"] for f in folds]
        print(f"CV по сессиям: средний MAE={np.mean(maes):.4f}, худший={np.max(maes):.4f} "
              f"(сессия '{folds[int(np.argmax(maes))]['session']}'), {time.time() - t0:.0f} с")
        json.dump(dict(mode="cv", arch=a.arch, pretrained=a.pretrained, aug=a.aug, epochs=a.epochs,
                       folds=folds, mean_mae=float(np.mean(maes)), worst_mae=float(np.max(maes))),
                  open(os.path.join(a.out, "cv-report.json"), "w"), indent=2, ensure_ascii=False)
        return

    # Обычный режим: одно разбиение.
    if a.val_frac <= 0:
        tr, va, val_names = np.arange(N), np.array([], int), []
    elif a.split == "sessions":
        if n_groups < 2:
            raise SystemExit("для --split sessions нужно минимум 2 сессии (или --split frames)")
        if a.val_groups:
            vg = []
            for s in (x.strip() for x in a.val_groups.split(",")):
                if s in gnames:
                    vg.append(gnames.index(s))
                elif s.isdigit() and int(s) < n_groups:
                    vg.append(int(s))
                else:
                    raise SystemExit(f"--val-groups: нет сессии «{s}»; есть: {', '.join(gnames)}")
        else:
            nv = max(1, round(a.val_frac * n_groups))
            vg = sorted(rng.choice(n_groups, nv, replace=False).tolist())
        va = np.flatnonzero(np.isin(groups, vg))
        tr = np.flatnonzero(~np.isin(groups, vg))
        val_names = [gnames[g] for g in vg]
    else:
        print("ВНИМАНИЕ: деление по кадрам — оценка завышена, кадры одной сессии попадают и в train, и в val")
        idx = rng.permutation(N)
        nv = int(N * a.val_frac)
        va, tr, val_names = idx[:nv], idx[nv:], ["(случайные кадры)"]
    print(f"train={len(tr)} кадров, val={len(va)} кадров; валидационные сессии: {val_names}")

    net, state, hist = train_once(a, X, Y, G, tr, va, K, perm, mirror, n_groups, dev)
    report = dict(mode="single", arch=a.arch, pretrained=a.pretrained, aug=a.aug, split=a.split,
                  val_sessions=val_names, epochs=hist)
    if len(va):
        vm, vch, vg = evaluate(net, X, Y, G, va, a.batch * 2, n_groups)
        print(f"лучший val MAE={vm:.4f}; по каналам: " + ", ".join(f"{c}={v:.3f}" for c, v in zip(meta["channels"], vch)))
        dirs = direction_scores(net, X, Y, va, a.batch * 2, meta["channels"])
        if dirs:
            print(format_directions(dirs))
            report["directions"] = dirs
        for g, v in sorted(vg.items()):
            print(f"  сессия '{gnames[g]}': MAE={v:.4f}")
        report.update(val_mae=vm, per_channel=dict(zip(meta["channels"], vch)),
                      per_session={gnames[g]: v for g, v in vg.items()})
    ck = os.path.join(a.out, "best.pt")
    torch.save({"state": state, "arch": a.arch, "channels": meta["channels"], "height": meta["height"],
                "width": meta["width"], "width_mult": a.width, "flip": meta.get("flip"),
                "pretrained": a.pretrained, "init": a.init}, ck)
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=2, ensure_ascii=False)
    print(f"чекпойнт → {ck}, отчёт → {os.path.join(a.out, 'report.json')}, {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Синтетический датасет рта с известной истиной — для проверки конвейера обучения
без шлема.

Рисует «рот» (тёмная полость + текстура кожи) с управляемыми параметрами-метками:
  ch0 openness — раскрытие (высота полости)
  ch1 widen    — ширина (улыбка)
  ch2 shiftX   — горизонтальный сдвиг полости (грубый аналог «язык вбок»)

Чтобы синтетика воспроизводила главную трудность реального трекинга, кадры
разбиты на **сессии**, и у каждой своя «посадка шлема»: сдвиг, масштаб, поворот и
яркость всей картинки. Плюс в кадре есть яркий неподвижный относительно лица
**ориентир** (как блик на носу у реальной камеры) — по нему модель может отличить
сдвиг шлема от сдвига полости. Без ориентира shiftX на новой посадке неотличим от
сдвига шлема.

Пишет frames.npy, labels.npy, groups.npy (номер сессии) и meta.json с правилами
отражения для аугментации. Это не лицо — хранить можно свободно.
"""
import argparse, json, os
import numpy as np

CHANNELS = ["openness", "widen", "shiftX"]
# При отражении кадра по горизонтали: openness/widen не меняются, shiftX → 1 - shiftX.
FLIP = {"swap": [], "mirror": ["shiftX"]}


def random_mount(rng, strength):
    """Посадка шлема на сессию: сдвиг, масштаб, поворот, яркость."""
    return dict(
        dx=rng.uniform(-0.06, 0.06) * strength,
        dy=rng.uniform(-0.06, 0.06) * strength,
        scale=1.0 + rng.uniform(-0.15, 0.15) * strength,
        rot=np.deg2rad(rng.uniform(-8, 8) * strength),
        gain=1.0 + rng.uniform(-0.15, 0.15) * strength,
    )


def render(h, w, openness, widen, shiftx, mount, anchor, rng):
    """Кадр uint8 [h,w]: каноническая сцена, видимая через посадку mount."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx0, cy0 = w / 2, h / 2
    # Обратное преобразование посадки: пиксель кадра → координата в «лице».
    x0 = xx - cx0 - mount["dx"] * w
    y0 = yy - cy0 - mount["dy"] * h
    c, s = np.cos(-mount["rot"]), np.sin(-mount["rot"])
    fx = (c * x0 - s * y0) / mount["scale"] + cx0
    fy = (s * x0 + c * y0) / mount["scale"] + cy0

    cx = w * (0.30 + 0.40 * shiftx)
    cy = h * 0.58
    rx = w * (0.16 + 0.20 * widen)
    ry = h * (0.04 + 0.24 * openness)
    d = ((fx - cx) / rx) ** 2 + ((fy - cy) / ry) ** 2
    skin = 150 + 30 * np.sin(fx * 0.05 + rng.uniform(0, 6.28))
    img = np.where(d <= 1.0, 40 + 40 * openness, skin)
    if anchor:   # яркий ориентир над ртом, неподвижный относительно лица
        da = ((fx - w * 0.5) / (w * 0.07)) ** 2 + ((fy - h * 0.14) / (h * 0.06)) ** 2
        img = np.where(da <= 1.0, 250.0, img)
    img = img * mount["gain"] + rng.normal(0, 3, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--sessions", type=int, default=10, help="число сессий (посадок шлема)")
    ap.add_argument("--mount", type=float, default=1.0, help="сила разброса посадок, 0 = все одинаковые")
    ap.add_argument("--no-anchor", action="store_true", help="без ориентира в кадре")
    ap.add_argument("--h", type=int, default=120)
    ap.add_argument("--w", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    mounts = [random_mount(rng, a.mount) for _ in range(a.sessions)]
    frames = np.empty((a.n, a.h, a.w), np.uint8)
    labels = np.empty((a.n, len(CHANNELS)), np.float32)
    groups = np.empty(a.n, np.int32)
    for i in range(a.n):
        g = i * a.sessions // a.n          # сессии идут подряд, как реальные записи
        o, wd, sx = rng.random(3)
        frames[i] = render(a.h, a.w, o, wd, sx, mounts[g], not a.no_anchor, rng)
        labels[i] = (o, wd, sx)
        groups[i] = g

    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "frames.npy"), frames)
    np.save(os.path.join(a.out, "labels.npy"), labels)
    np.save(os.path.join(a.out, "groups.npy"), groups)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"height": a.h, "width": a.w, "channels": CHANNELS, "flip": FLIP}, f, indent=2)
    print(f"записано {a.n} кадров {a.w}x{a.h}, сессий={a.sessions}, разброс посадок={a.mount}, "
          f"ориентир={'нет' if a.no_anchor else 'да'} → {a.out}")


if __name__ == "__main__":
    main()

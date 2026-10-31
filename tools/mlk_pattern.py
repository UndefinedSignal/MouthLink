#!/usr/bin/env python3
r"""Тест аватара без шлема и модели: шлёт в модуль VRCFT заданные позы языка (MLK2).

Позволяет отделить проблемы аватара от проблем модели: если на «вправо до упора» язык на аватаре
не уходит вправо или пропадает, дело в аватаре или его анимациях, а не в распознавании.
Остальные каналы рта не трогаются (идут от стока). mouthlink_live.py в это время не запускать —
оба шлют на один и тот же порт.

  python tools/mlk_pattern.py                 # круг: прямо, влево, вправо, вниз, диагонали, плавно
  python tools/mlk_pattern.py --hold 4 --loops 3
  python tools/mlk_pattern.py --pose right    # держать одну позу (Ctrl+C — выход)

Позы: center, left, right, down, left_down, right_down, sweep (плавно влево-вправо), in (язык убран).
"""
import argparse, math, socket, struct, time

POSES = {
    "in":         dict(TongueOut=0.0),
    "center":     dict(TongueOut=1.0),
    "left":       dict(TongueOut=1.0, TongueLeft=1.0),
    "right":      dict(TongueOut=1.0, TongueRight=1.0),
    "down":       dict(TongueOut=1.0, TongueDown=1.0),
    "left_down":  dict(TongueOut=1.0, TongueLeft=0.7, TongueDown=0.7),
    "right_down": dict(TongueOut=1.0, TongueRight=0.7, TongueDown=0.7),
}
ORDER = ["center", "left", "right", "down", "left_down", "right_down", "sweep", "in"]
CHANNELS = ["TongueOut", "TongueLeft", "TongueRight", "TongueUp", "TongueDown"]
NAMES = ",".join(CHANNELS).encode()


def packet(seq, vals):
    v = [float(vals.get(c, 0.0)) for c in CHANNELS]
    return (struct.pack("<4sHHIQH", b"MLK2", 2, len(CHANNELS), seq, time.monotonic_ns(), len(NAMES))
            + NAMES + struct.pack(f"<{len(v)}f", *v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="127.0.0.1:9050")
    ap.add_argument("--hold", type=float, default=3.0, help="секунд на позу")
    ap.add_argument("--loops", type=int, default=1, help="сколько кругов (0 — бесконечно)")
    ap.add_argument("--pose", choices=list(POSES) + ["sweep"], help="держать одну позу")
    a = ap.parse_args()
    host, port = a.out.rsplit(":", 1)
    dst = (host, int(port))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    period = 1 / 90

    def run(name, seconds):
        nonlocal seq
        print(f"  {name}", flush=True)
        t0 = time.monotonic()
        while seconds is None or time.monotonic() - t0 < seconds:
            t = time.monotonic() - t0
            if name == "sweep":
                x = math.sin(2 * math.pi * t / 3.0)          # влево-вправо за 3 с
                vals = dict(TongueOut=1.0, TongueLeft=max(0.0, -x), TongueRight=max(0.0, x))
            else:
                vals = POSES[name]
            seq = (seq + 1) & 0xFFFFFFFF
            s.sendto(packet(seq, vals), dst)
            time.sleep(period)

    print(f"позы → {dst[0]}:{dst[1]} (модуль UnifiedPicoModule-MouthLink). Ctrl+C — выход.")
    try:
        if a.pose:
            run(a.pose, None)
        loop = 0
        while a.loops == 0 or loop < a.loops:
            for name in ORDER:
                run(name, a.hold * (2 if name == "sweep" else 1))
            loop += 1
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(5):                                  # убрать язык
            seq = (seq + 1) & 0xFFFFFFFF
            s.sendto(packet(seq, POSES["in"]), dst)
        s.close()


if __name__ == "__main__":
    main()

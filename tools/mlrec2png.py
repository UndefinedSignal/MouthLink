#!/usr/bin/env python3
"""Разложить записанную сессию .mlrec в PNG-кадры для просмотра офлайн.

Работает и с записями эмулятора, и с реальными записями со шлема.
Понимает только несжатые GRAY8/GRAY16LE (compression=0). Для lz4/jpeg
сначала распакуй на стороне C# или допиши сюда декод.

Использование:
    python3 mlrec2png.py session.mlrec out_dir [--max N] [--stride K]

--max N   — не больше N кадров
--stride K — брать каждый K-й кадр (прореживание)

Кадры — снимки лица: держи выгрузку только локально.
"""
import struct, sys, os, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec"); ap.add_argument("out")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    try:
        from PIL import Image
    except ImportError:
        sys.exit("нужен Pillow: pip install pillow")

    os.makedirs(a.out, exist_ok=True)
    with open(a.rec, "rb") as f:
        if f.read(4) != b"MLR1":
            sys.exit("не похоже на .mlrec (нет магии MLR1)")
        ver, _ = struct.unpack("<HH", f.read(4))
        n = saved = 0
        while True:
            lenb = f.read(4)
            if len(lenb) < 4:
                break
            rec_len = struct.unpack("<I", lenb)[0]
            rec = f.read(rec_len)
            if len(rec) < rec_len:
                break
            frame_id, cap_ns = struct.unpack("<IQ", rec[0:12])
            pixfmt, comp = rec[12], rec[13]
            w, h, rx, ry = struct.unpack("<HHHH", rec[14:22])
            plen = struct.unpack("<I", rec[22:26])[0]
            payload = rec[26:26 + plen]

            if n % a.stride == 0 and comp == 0:
                if pixfmt == 0x01 and len(payload) >= w * h:          # GRAY8
                    img = Image.frombytes("L", (w, h), payload[:w * h])
                elif pixfmt == 0x02 and len(payload) >= w * h * 2:    # GRAY16LE → 8 бит для глаза
                    import array
                    a16 = array.array("H"); a16.frombytes(payload[:w * h * 2])
                    mx = max(a16) or 1
                    img = Image.frombytes("L", (w, h),
                        bytes(min(255, v * 255 // mx) for v in a16))
                else:
                    n += 1; continue
                img.save(os.path.join(a.out, f"f{frame_id:08d}.png"))
                saved += 1
            n += 1
            if a.max and saved >= a.max:
                break
    print(f"прочитано записей: {n}, сохранено PNG: {saved} → {a.out}")

if __name__ == "__main__":
    main()

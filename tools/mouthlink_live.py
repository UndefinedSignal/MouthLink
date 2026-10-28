#!/usr/bin/env python3
r"""MouthLink live: кадры камеры рта со шлема → ONNX-модель → направление языка в VRChat.

Подписывается на шлем (модуль PicoET-Enhance-MouthLink), на каждом кадре (90 Гц) прогоняет
модель, сглаживает направления языка и шлёт их в UnifiedPicoModule (сборка с MouthLink
tongue) на 127.0.0.1:9050. Модуль включает TongueLeft/Right/Up/Down только пока стоковый
трекинг видит высунутый язык.

  python tools/mouthlink_live.py --host <IP-шлема> --onnx training\runs\final\mouth.onnx

Важно: шлем отдаёт поток одному подписчику — MouthLink.Receiver в это время не запускать.
Нужно: pip install onnxruntime pillow numpy (для GPU — onnxruntime-gpu / onnxruntime-directml).

Опции:
  --lo 0.15 --hi 0.55   выход модели ≤ lo → 0, ≥ hi → 1 (модель осторожная: позитивы часто < 0.5)
  --tau 0.04            постоянная сглаживания, с (0 — без сглаживания)
  --meta FILE           описание модели (.json от export_onnx.py или meta.json датасета),
                        если рядом с .onnx нет <имя>.json
  --out 127.0.0.1:9050  куда слать MLK1
  --cpu                 только CPU
"""
import argparse, json, os, socket, struct, sys, threading, time, zlib

import numpy as np

DIRS = ["TongueLeft", "TongueRight", "TongueUp", "TongueDown"]
HDR = 42


class Reassembler:
    """Сборка FRAGMENT-датаграмм MouthLink в кадры (как MouthLink.Core.FrameReassembler)."""

    def __init__(self, keep=8):
        self.pending = {}
        self.keep = keep
        self.bad = self.dropped = self.completed = 0

    def feed(self, d):
        if len(d) < HDR or d[0] != 0x01 or d[1] != 1:
            return None
        if zlib.crc32(d[:38]) & 0xFFFFFFFF != struct.unpack_from("<I", d, 38)[0]:
            self.bad += 1
            return None
        fmt, comp = d[2], d[3]
        fid, cap, w, h, rx, ry, plen, idx, cnt, off, flen = struct.unpack_from("<IQHHHHIHHIH", d, 4)
        if cnt == 0 or idx >= cnt or plen == 0 or off + flen > plen or len(d) < HDR + flen:
            self.bad += 1
            return None
        p = self.pending.get(fid)
        if p is None:
            if len(self.pending) >= self.keep:
                oldest = min(self.pending)
                del self.pending[oldest]
                self.dropped += 1
            p = self.pending[fid] = {"buf": bytearray(plen), "got": set(), "cnt": cnt}
        if idx in p["got"]:
            return None
        p["buf"][off:off + flen] = d[HDR:HDR + flen]
        p["got"].add(idx)
        if len(p["got"]) < p["cnt"]:
            return None
        del self.pending[fid]
        for old in [k for k in self.pending if (fid - k) & 0xFFFFFFFF < 0x80000000]:
            del self.pending[old]          # старше готового — уже не нужны
            self.dropped += 1
        self.completed += 1
        return dict(fid=fid, cap=cap, fmt=fmt, comp=comp, w=w, h=h, x=rx, y=ry, payload=bytes(p["buf"]))


def subscribe_packet(listen_port):
    # SUBSCRIBE v1, 16 байт: fps 0 (каждый кадр), формат на выбор источника, без ROI, без STOCK.
    return struct.pack("<BBHBBHHHHBB", 0x02, 1, listen_port, 0, 0, 0, 0, 0, 0, 0, 0)


class Source(threading.Thread):
    def __init__(self, host, ctrl, listen):
        super().__init__(daemon=True)
        self.dst = (host, ctrl)
        self.listen = listen
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        self.sock.bind(("0.0.0.0", listen))
        self.sock.settimeout(0.1)
        self.ra = Reassembler()
        self.lock = threading.Lock()
        self.new = threading.Event()
        self.latest = None
        self.frames = 0
        self.stop = False

    def run(self):
        sub = subscribe_packet(self.listen)
        next_sub = 0.0
        while not self.stop:
            now = time.monotonic()
            if now >= next_sub:
                try:
                    self.sock.sendto(sub, self.dst)
                except OSError:
                    pass
                next_sub = now + 0.5
            try:
                d, _ = self.sock.recvfrom(65536)
            except (socket.timeout, ConnectionResetError):
                continue
            except OSError:
                if self.stop:
                    break
                continue
            f = self.ra.feed(d)
            if f is not None:
                with self.lock:
                    self.latest = f
                    self.frames += 1
                self.new.set()

    def close(self):
        self.stop = True
        try:
            self.sock.sendto(bytes([0x03, 1]), self.dst)   # UNSUBSCRIBE
        except OSError:
            pass


def load_meta(onnx_path, meta_path):
    cands = [meta_path] if meta_path else [os.path.splitext(onnx_path)[0] + ".json", onnx_path + ".json"]
    for c in cands:
        if c and os.path.exists(c):
            m = json.load(open(c, encoding="utf-8"))
            if "channels" in m:
                return m, c
    raise SystemExit("нет описания каналов модели: перевыгрузи ONNX свежим export_onnx.py "
                     "(он пишет <имя>.json) или укажи --meta путь\\к\\meta.json датасета")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--meta")
    ap.add_argument("--ctrl", type=int, default=9040)
    ap.add_argument("--listen", type=int, default=9041)
    ap.add_argument("--out", default="127.0.0.1:9050")
    ap.add_argument("--lo", type=float, default=0.15)
    ap.add_argument("--hi", type=float, default=0.55)
    ap.add_argument("--tau", type=float, default=0.04)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seconds", type=float, default=0, help="остановиться через N секунд (для тестов)")
    a = ap.parse_args()
    if not 0 <= a.lo < a.hi <= 1:
        raise SystemExit("нужно 0 ≤ --lo < --hi ≤ 1")

    import onnxruntime as ort
    from PIL import Image

    meta, meta_src = load_meta(a.onnx, a.meta)
    channels = meta["channels"]
    missing = [d for d in DIRS if d not in channels]
    if missing:
        raise SystemExit(f"в модели нет каналов {missing}: нужна модель, обученная с mlrec_pair --tongue")
    i_dirs = [channels.index(d) for d in DIRS]
    i_out = channels.index("TongueOut") if "TongueOut" in channels else None

    avail = ort.get_available_providers()
    want = ["CPUExecutionProvider"] if a.cpu else [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider",
                                                              "CPUExecutionProvider") if p in avail]
    sess = ort.InferenceSession(a.onnx, providers=want)
    inp = sess.get_inputs()[0]
    H, W = int(inp.shape[2]), int(inp.shape[3])
    print(f"модель: {a.onnx} ({len(channels)} каналов, вход {W}x{H}, {sess.get_providers()[0]}); описание: {meta_src}")
    print(f"направления → {a.out}; lo={a.lo} hi={a.hi} tau={a.tau}s. Ctrl+C — выход.")

    oh, op = a.out.rsplit(":", 1)
    out_addr = (oh, int(op))
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    src = Source(a.host, a.ctrl, a.listen)
    src.start()

    smooth = np.zeros(4, np.float32)
    last_cap = None
    seq = 0
    t0 = time.monotonic()
    stat_t = t0
    n_inf = 0
    inf_ms = 0.0
    frames_at = 0
    last_frame_wall = t0
    warned_roi = warned_fmt = hinted = False
    try:
        while True:
            if a.seconds and time.monotonic() - t0 > a.seconds:
                break
            if not src.new.wait(0.25):
                if time.monotonic() - last_frame_wall > 3 and not hinted:
                    print("кадров нет: надень шлем, проверь бридж/VRCFT и что MouthLink.Receiver не запущен")
                    hinted = True
                continue
            src.new.clear()
            with src.lock:
                f = src.latest
            if f is None or f["cap"] == last_cap:
                continue
            last_frame_wall = time.monotonic()
            hinted = False
            if f["fmt"] != 1 or f["comp"] != 0:
                if not warned_fmt:
                    print(f"пропускаю кадры формата {f['fmt']}/{f['comp']} (нужен GRAY8 без сжатия)")
                    warned_fmt = True
                continue
            if (f["x"] or f["y"]) and not warned_roi:
                print("ВНИМАНИЕ: шлем шлёт ROI, а модель обучена на полном кадре")
                warned_roi = True
            t1 = time.perf_counter()
            img = np.frombuffer(f["payload"], np.uint8).reshape(f["h"], f["w"])
            if img.shape != (H, W):
                img = np.asarray(Image.fromarray(img).resize((W, H), Image.BILINEAR))  # как mlrec_pair --size
            x = (img.astype(np.float32) / 255.0)[None, None]
            y = sess.run(None, {inp.name: x})[0][0]
            inf_ms += (time.perf_counter() - t1) * 1000
            n_inf += 1

            raw = y[i_dirs].astype(np.float32)
            dt = 1 / 90 if last_cap is None else max(0.0, min(0.2, (f["cap"] - last_cap) / 1e9))
            last_cap = f["cap"]
            alpha = 1.0 if a.tau <= 0 else 1.0 - np.exp(-dt / a.tau)
            smooth += (raw - smooth) * alpha
            mapped = np.clip((smooth - a.lo) / (a.hi - a.lo), 0, 1)
            tout = float(np.clip(y[i_out], 0, 1)) if i_out is not None else 0.0
            seq = (seq + 1) & 0xFFFFFFFF
            out.sendto(struct.pack("<4sHHIQ5f", b"MLK1", 1, 5, seq, f["cap"], *mapped.tolist(), tout), out_addr)

            now = time.monotonic()
            if now - stat_t >= 1.0:
                with src.lock:
                    fr = src.frames
                rx = (fr - frames_at) / (now - stat_t)
                frames_at = fr
                bars = "  ".join(f"{d[6:]:<5}{'#' * int(round(v * 10)):<10}{v:.2f}" for d, v in zip(DIRS, mapped))
                print(f"кадры {rx:5.1f}/с  модель {n_inf / (now - stat_t):5.1f}/с {inf_ms / max(1, n_inf):4.1f} мс  "
                      f"TongueOut {tout:.2f}  |  {bars}")
                stat_t, n_inf, inf_ms = now, 0, 0.0
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        out.close()
        print(f"остановлено: кадров собрано {src.ra.completed}, битых фрагментов {src.ra.bad}, неполных {src.ra.dropped}")


if __name__ == "__main__":
    main()

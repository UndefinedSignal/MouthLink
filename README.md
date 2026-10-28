# MouthLink — custom mouth tracking for PICO 4 Pro

**English** | [Русский](README.ru.md)

Custom mouth tracking for the PICO 4 Pro. Frames from the headset's mouth camera are
streamed to the PC, where your own model turns them into blendshapes for VRChat via
VRCFaceTracking. The goal is what stock tracking cannot do: personal calibration for
your face and channels the stock model does not have at all — first of all **tongue
direction** (left / right / up / down; later circular motion). Stock PICO only reports
a single `TongueOut` value.

Eyes stay on the stock path (PicoET-Enhance → PicoFacialBridge → UnifiedPicoModule):
90 Hz, real pupils, independent eyes. Only the mouth is custom.

## How it works

```
   HEADSET (PICO 4 Pro, rooted)                      PC
 ┌───────────────────────────────────┐   ┌───────────────────────────────────┐
 │ pxreyetrackingservice             │   │ MouthLink.Receiver                │
 │  └ PicoET-Enhance-MouthLink       │   │   reassembly, stats, live view,   │
 │     ├ mouth camera frame ─────────┼──▶│   .mlrec recording,               │
 │     │  (400×400 GRAY8, 90 Hz)     │UDP│   stock labels (.stock.csv),      │
 │     └ stock mouth values ─────────┼──▶│   voice prompts (.prompts.csv)    │
 │        (same timestamp)           │   │        │                          │
 └───────────────────────────────────┘   │        ▼                          │
                                         │ tools/mlrec_pair.py → dataset     │
                                         │ training/ → model (ONNX)          │
                                         │        │                          │
                                         │        ▼ (planned) VRCFT module   │
                                         └───────────────────────────────────┘
```

- **Headset side** — the *PicoET-Enhance-MouthLink* fork of
  [PicoET-Enhance](https://github.com/WolalaQAQ/PicoET-Enhance) (branch `mouthlink`).
  Capture happens inside the payload PicoET already injects (no second injector, no
  extra camera client). Opt-in (`picoet.sh mouthlink-on`), nothing is sent before a PC
  subscribes. For every mouth frame it can also send the stock values with the **same
  source timestamp**, so frames and labels pair exactly.
- **PC side** — this repository: protocol, receiver, recording and playback, labelling,
  dataset builder and training.

## Verified on hardware (PICO 4 Pro, PICO OS 5.13.3S)

- Mouth camera: **400×400, 8-bit IR, 90 fps**; full stream ≈118 Mbit/s uncompressed,
  ~20 ms latency, practically lossless alongside Virtual Desktop; stock eye/mouth
  tracking stays at 90/90 Hz.
- Stock labels: 100 % of frames paired with stock values across all sessions.
- Stock distillation: 8 sessions (≈44k frames, 160×160), MobileNetV3-small, evaluated
  on unseen headset fits (leave-one-session-out): **mean MAE 0.034**, worst session 0.049.
- The mouth camera looks at the face **from the side**, so horizontal-flip augmentation
  is off by default.

## Repository

| Path | What it is |
|------|------------|
| `PROTOCOL.md` | Wire protocol: UDP fragments with CRC32, SUBSCRIBE (fps, ROI, flags), PING/PONG clock sync, DISCOVER/HELLO, STOCK labels |
| `MouthLink.Core` | Framing, reassembly, `.mlrec` sessions, STOCK packet + CSV log |
| `MouthLink.Receiver` | Subscription, statistics, recording, browser live view, ROI, stock logging, voice-prompted calibration |
| `MouthLink.Emulator` | Synthetic frame source for working without a headset (no ROI/STOCK) |
| `MouthLink.Player` | Offline `.mlrec` player in the browser |
| `tools/mlrec_check.py` | Recording health check: rate, gaps, frozen/black/overexposed frames |
| `tools/mlrec2png.py` | Export a recording to PNG |
| `tools/mlrec_pair.py` | Frames + stock (+ prompts) → training dataset |
| `training/` | PyTorch training (tiny CNN / MobileNetV3), masked loss for partial labels, session-wise CV, ONNX export |

Build: .NET 8 SDK (`dotnet build` in the solution root). Python 3 with NumPy for the
tools; PyTorch (+ Pillow for resizing) for training.

## Usage with the headset

Requirements: rooted PICO 4 Pro with the PicoET-Enhance-MouthLink module (ml3 or newer)
in `dual` mode with `mouthlink-on`; PicoFacialBridge started and **VRCFaceTracking with
UnifiedPicoModule running** (the bridge only starts tracking — and the cameras — when a
PC client is connected).

**Record a session** (all camera frames + stock labels):
```
dotnet run --project MouthLink.Receiver -- --host <headset-IP> --fps 0 --record d:\mouthrecon\s01.mlrec
```
Live view: `http://localhost:8080/`. `--fps 0` means every camera frame (90 fps);
`--roi x,y,w,h` makes the headset send only a rectangle; `--no-stock` disables labels.
Ctrl+C stops cleanly. Take the headset off between sessions: each recording is one
headset fit, and models are evaluated on fits they have not seen.

**Tongue calibration with voice prompts:**
```
dotnet run --project MouthLink.Receiver -- --host <headset-IP> --fps 0 --record d:\mouthrecon\t01.mlrec --prompts tongue
```
The receiver says "tongue left / right / up / down / straight" (Russian voice if
installed in Windows, otherwise English; `--voice en|off`), 3 shuffled rounds of 4 s,
≈2 min, and stops by itself. Left/right are **your own** sides.

**Build a dataset and train:**
```
python tools/mlrec_pair.py d:\mouthrecon\data\real "d:\mouthrecon\[st]*.mlrec" --tongue --size 160 --every 2
cd training
python train.py --data d:\mouthrecon\data\real --out runs\real --arch mnv3 --pretrained --cv
```
`mlrec_pair.py` pairs frames and stock values by exact timestamp, uses the unsmoothed
stock values as labels (no filter lag), and with `--tongue` adds `TongueLeft/Right/Up/Down`:
taken from the prompts inside prompted intervals, 0 where stock says the tongue is in,
unknown (NaN, ignored by the loss) elsewhere. For direction channels `train.py` reports
"recognised X % / false Y %" — plain MAE is meaningless for such rare channels.
See `training/README.md` (Russian) for details.

## Without a headset

```
dotnet run --project MouthLink.Emulator -- 9040 400 400 90 --pattern
dotnet run --project MouthLink.Receiver -- --host 127.0.0.1 --verify --record pattern.mlrec
python tools/mlrec_check.py pattern.mlrec --pattern
```
`--pattern` / `--verify` check every frame pixel by pixel. The training pipeline can be
exercised on synthetic data with `training/make_synth_dataset.py`.

## Privacy

Mouth-camera frames are images of your face. Recordings, stock and prompt logs, PNG
exports, datasets and model weights stay local: `.gitignore` excludes `*.mlrec`,
`*.stock.csv`, `*.prompts.csv`, `*.npy`, `*.onnx`, `*.pt`, `data/`, `runs/`. Only code
goes into the repository. The live view and player listen on loopback only. The UDP
stream has no authentication — use it on a trusted LAN only.

## Roadmap

- Tongue direction from prompted sessions (in progress), then circular motion.
- VRCFaceTracking module: ONNX inference on the PC, merged with stock (stock as a floor/gate).
- Optional compression (lz4) and an address allowlist on the headset side.

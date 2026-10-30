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
| `tools/mlrec_eval.py` | Per-pose check of a model on a prompted recording |
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

**Full-face calibration (all expressions):**
```
dotnet run --project MouthLink.Receiver -- --host <headset-IP> --fps 0 --record d:\mouthrecon\f01.mlrec --prompts face
```
72 prompts from `prompts/face.json` in 10 groups — jaw, lips, mouth corners, nose,
cheeks, tongue, tongue with the mouth wide open / almost closed (`tongue_jaw`), open mouth
with the tongue inside (`mouth_open`, teaches "no tongue"), emotions, speech — shuffled, each followed by a short "relax" pause.
Simple poses are held 4 s; one-sided ones (left/right cheek, corner, tightener) 6 s with
a longer settle, because they take a moment to find; emotions 5 s, speech 9–12 s.
One round ≈ 9.5 min. Options: `--skip tongue_roll,...` (expressions you cannot do),
`--groups jaw,lips,...`, `--rounds N`, `--time-scale 1.5` (everything slower),
`--catalog my.json` (own catalog). Tongue only, without "up":
`--groups tongue,tongue_jaw,mouth_open --skip tongue_up,tongue_roll --rounds 3` (≈ 7 min). The catalog adds channels the stock model lacks:
`CheekPuffLeft/Right`, `CheekSuckLeft/Right`, `MouthTightenerLeft/Right`, `TongueRoll`.

**Build a dataset and train:**
```
python tools/mlrec_pair.py d:\mouthrecon\data\real "d:\mouthrecon\[st]*.mlrec" --tongue --size 160 --every 2
cd training
python train.py --data d:\mouthrecon\data\real --out runs\real --arch mnv3 --pretrained --cv
```
New recordings later: pair only them into a separate dataset and fine-tune the existing model
(`--data old,new --init runs\...\best.pt`) — see "Дообучение" in `training/README.md`.
For face sessions use `--face` instead of `--tongue` (all sessions can be mixed —
tongue prompts are understood too):
```
python tools/mlrec_pair.py d:\mouthrecon\data\face "d:\mouthrecon\[stf]*.mlrec" --face --size 160 --every 2
```
With `--face`, inside a prompt (after its settle time) the catalog targets override
stock — e.g. "smile left" sets `MouthSmileL` to 1 and `MouthSmileR` to 0; new channels
not named by the prompt become 0; emotions and speech keep stock labels. The report
lists prompts where stock disagrees with the target — that is exactly what the own
model learns beyond stock.

`mlrec_pair.py` pairs frames and stock values by exact timestamp, uses the unsmoothed
stock values as labels (no filter lag), and with `--tongue` adds `TongueLeft/Right/Up/Down`:
taken from the prompts inside prompted intervals, 0 where stock says the tongue is in,
unknown (NaN, ignored by the loss) elsewhere. For direction channels `train.py` reports
"recognised X % / false Y %" — plain MAE is meaningless for such rare channels.
See `training/README.md` (Russian) for details.

**Check a model pose by pose** — best on a fresh session the model has not seen:
```
python tools/mlrec_eval.py --onnx training\runs\face-final\mouth.onnx d:\mouthrecon\f09.mlrec
```
For every prompt it prints target, stock and raw model output per channel, the hit rate,
which other channels the model lights up, and false positives on "relax".

## Live in VRChat

1. Train the final model on all sessions and export it (writes `mouth.onnx` + `mouth.json`):
   ```
   cd training
   python train.py --data d:\mouthrecon\data\face --out runs\face-final --arch mnv3 --pretrained --val-frac 0
   python export_onnx.py --ckpt runs\face-final\best.pt --out runs\face-final\mouth.onnx
   ```
2. Install the *UnifiedPicoModule-MouthLink* build of UnifiedPicoModule (VRCFT closed while
   replacing the DLL). It listens on `127.0.0.1:9050` and lays the model values over stock.
3. Bridge, VRCFaceTracking and VRChat as usual; **MouthLink.Receiver must not run** (the
   headset serves one subscriber). Then:
   ```
   pip install onnxruntime pillow numpy
   python tools/mouthlink_live.py --host <headset-IP> --onnx training\runs\face-final\mouth.onnx
   ```
   It infers every frame (90 fps), smooths (`--tau`) and sends all model channels by name
   (MLK2). Channels stock lacks (tongue directions, separate cheeks, lip tighteners,
   TongueRoll) are stretched with `--lo/--hi`; stock channels are sent as predicted. Once a
   second it prints the tongue directions and the most active channels. `--attack 0.15`
   makes the tongue rise slower than it falls, so a short false spike when the mouth opens
   does not throw the tongue out (costs ~0.15 s of delay when it really comes out).
4. Per channel, the module decides what to use — `MouthLink.json` next to its DLL
   (`%APPDATA%\VRCFaceTracking\CustomLibs\<module>\`), re-read within a second of saving:
   `"model"` (default), `"stock"`, `"blend:0.5"`, `"max"`, e.g.
   `"Modes": { "JawOpen": "stock", "MouthSmileL": "blend:0.5" }`. Tongue directions and
   TongueRoll show only while TongueOut is up. Without fresh model values (script stopped)
   everything falls back to stock within 150 ms. Eye tracking is never touched.
   The older tongue-only format is still available: `--proto 1`.

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
- Full face live in VRChat: per-channel tuning of model vs stock on real sessions.
- Optional compression (lz4) and an address allowlist on the headset side.

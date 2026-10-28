using System;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MouthLink.Core;

namespace MouthLink.Receiver;

/// <summary>
/// Живой просмотр без GUI-зависимостей: маленький HTTP-сервер отдаёт последний
/// собранный кадр сырыми байтами, а рисует его страница в браузере на canvas.
/// Открой http://localhost:&lt;port&gt;/ в браузере.
///
/// Кадр — снимок лица: сервер слушает только loopback, наружу ничего не отдаёт.
/// </summary>
public sealed class LiveViewServer : IDisposable
{
    private readonly HttpListener _http = new();
    private readonly int _port;
    private volatile Snapshot? _latest;
    private long _published;

    private sealed class Snapshot
    {
        public required byte[] Header;   // 12-байтовая шапка (см. Publish)
        public required byte[] Payload;  // кадр как на проводе
    }

    public LiveViewServer(int port)
    {
        _port = port;
        // Только loopback. На Windows префикс localhost разрешён без админ-прав,
        // а явный 127.0.0.1 часто требует netsh-резервирования и падает с access denied.
        _http.Prefixes.Add($"http://localhost:{port}/");
    }

    public void Start()
    {
        _http.Start();
        _ = Task.Run(Loop);
        Console.WriteLine($"живой просмотр: http://localhost:{_port}/");
    }

    /// <summary>Положить последний кадр. Дёшево: только упаковка шапки и подмена ссылки.</summary>
    public void Publish(AssembledFrame f)
    {
        var header = new byte[12];
        // u16 width, u16 height, u8 pixfmt, u8 compression, u32 frameId, u16 reserved
        header[0] = (byte)(f.Width & 0xFF); header[1] = (byte)(f.Width >> 8);
        header[2] = (byte)(f.Height & 0xFF); header[3] = (byte)(f.Height >> 8);
        header[4] = (byte)f.PixFmt;
        header[5] = (byte)f.Compression;
        header[6] = (byte)(f.FrameId & 0xFF); header[7] = (byte)((f.FrameId >> 8) & 0xFF);
        header[8] = (byte)((f.FrameId >> 16) & 0xFF); header[9] = (byte)((f.FrameId >> 24) & 0xFF);
        _latest = new Snapshot { Header = header, Payload = f.Payload };
        Interlocked.Increment(ref _published);
    }

    private async Task Loop()
    {
        while (_http.IsListening)
        {
            HttpListenerContext ctx;
            try { ctx = await _http.GetContextAsync(); }
            catch { return; }
            _ = Task.Run(() => Handle(ctx));
        }
    }

    private void Handle(HttpListenerContext ctx)
    {
        try
        {
            string path = ctx.Request.Url?.AbsolutePath ?? "/";
            switch (path)
            {
                case "/":
                    WriteText(ctx, "text/html; charset=utf-8", Html);
                    break;

                case "/frame.bin":
                {
                    var snap = _latest;
                    if (snap == null) { ctx.Response.StatusCode = 204; ctx.Response.Close(); break; }
                    ctx.Response.ContentType = "application/octet-stream";
                    ctx.Response.Headers["Cache-Control"] = "no-store";
                    ctx.Response.ContentLength64 = snap.Header.Length + snap.Payload.Length;
                    ctx.Response.OutputStream.Write(snap.Header, 0, snap.Header.Length);
                    ctx.Response.OutputStream.Write(snap.Payload, 0, snap.Payload.Length);
                    ctx.Response.OutputStream.Close();
                    break;
                }

                default:
                    ctx.Response.StatusCode = 404;
                    ctx.Response.Close();
                    break;
            }
        }
        catch
        {
            try { ctx.Response.Abort(); } catch { }
        }
    }

    private static void WriteText(HttpListenerContext ctx, string contentType, string body)
    {
        var bytes = Encoding.UTF8.GetBytes(body);
        ctx.Response.ContentType = contentType;
        ctx.Response.ContentLength64 = bytes.Length;
        ctx.Response.OutputStream.Write(bytes, 0, bytes.Length);
        ctx.Response.OutputStream.Close();
    }

    public void Dispose()
    {
        try { _http.Stop(); } catch { }
        try { _http.Close(); } catch { }
    }

    // Страница просмотра. Тянет /frame.bin в цикле requestAnimationFrame и рисует
    // на canvas. GRAY8 разворачивается в RGBA, GRAY16LE нормируется по максимуму,
    // JPEG (pixfmt 0x10) декодируется через createImageBitmap.
    private const string Html = """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MouthLink — живой просмотр</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #111; color: #ddd; font: 14px system-ui, sans-serif;
         display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 16px; }
  canvas { image-rendering: pixelated; background: #000; max-width: 95vw;
           border: 1px solid #333; border-radius: 6px; }
  #stat { font-variant-numeric: tabular-nums; color: #9bd; }
  label { color: #aaa; }
</style>
</head>
<body>
  <div id="stat">ожидание кадров…</div>
  <canvas id="cv" width="320" height="240"></canvas>
  <div>
    <label><input type="checkbox" id="fit" checked> растянуть ×2</label>
  </div>
<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const stat = document.getElementById('stat');
const fit = document.getElementById('fit');
let shown = 0, lastT = performance.now(), fps = 0, lastId = -1;

function applyScale(w, h) {
  const s = fit.checked ? 2 : 1;
  cv.style.width = (w * s) + 'px';
  cv.style.height = (h * s) + 'px';
}

async function drawGray(buf, w, h, bytesPerPx) {
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; applyScale(w, h); }
  const img = ctx.createImageData(w, h);
  const out = img.data;
  const n = w * h;
  if (bytesPerPx === 1) {
    for (let i = 0; i < n; i++) {
      const v = buf[i], j = i << 2;
      out[j] = v; out[j+1] = v; out[j+2] = v; out[j+3] = 255;
    }
  } else { // 16-бит LE, нормируем по максимуму кадра
    let mx = 1;
    for (let i = 0; i < n; i++) { const v = buf[i*2] | (buf[i*2+1] << 8); if (v > mx) mx = v; }
    for (let i = 0; i < n; i++) {
      const v = buf[i*2] | (buf[i*2+1] << 8);
      const g = (v * 255 / mx) | 0, j = i << 2;
      out[j] = g; out[j+1] = g; out[j+2] = g; out[j+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

async function drawJpeg(bytes, w, h) {
  const bmp = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; applyScale(w, h); }
  ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close();
}

async function tick() {
  try {
    const r = await fetch('/frame.bin', { cache: 'no-store' });
    if (r.status === 200) {
      const ab = await r.arrayBuffer();
      const dv = new DataView(ab);
      const w = dv.getUint16(0, true), h = dv.getUint16(2, true);
      const pixfmt = dv.getUint8(4);
      const frameId = dv.getUint32(6, true);
      const payload = new Uint8Array(ab, 12);
      if (pixfmt === 0x10) await drawJpeg(payload, w, h);
      else if (pixfmt === 0x02) await drawGray(payload, w, h, 2);
      else await drawGray(payload, w, h, 1);

      if (frameId !== lastId) { shown++; lastId = frameId; }
      const now = performance.now();
      if (now - lastT >= 500) { fps = shown * 1000 / (now - lastT); shown = 0; lastT = now; }
      stat.textContent = `${w}×${h}  fmt=0x${pixfmt.toString(16)}  кадр #${frameId}  ~${fps.toFixed(0)} fps`;
    }
  } catch (e) { stat.textContent = 'нет связи с приёмником: ' + e; }
  requestAnimationFrame(tick);
}
fit.addEventListener('change', () => applyScale(cv.width, cv.height));
requestAnimationFrame(tick);
</script>
</body>
</html>
""";
}

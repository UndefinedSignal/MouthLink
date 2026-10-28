using System;
using System.Globalization;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using MouthLink.Core;

// Офлайн-плеер записей .mlrec. Грузит файл с диска и показывает его в браузере
// с управлением: пауза, шаг, перемотка, скорость. Это просмотр файла, который
// уже есть на диске, — по сети кадры никуда не уходят, сервер слушает loopback.
//
// Запуск:  dotnet run --project MouthLink.Player -- session.mlrec [port]
// Потом открой http://localhost:<port>/ (по умолчанию 8081).

if (args.Length < 1)
{
    Console.Error.WriteLine("использование: MouthLink.Player <session.mlrec> [port]");
    return 1;
}
string path = args[0];
int port = args.Length > 1 ? int.Parse(args[1]) : 8081;

Console.WriteLine($"читаю {path} …");
SessionReader reader;
try { reader = new SessionReader(path); }
catch (Exception e) { Console.Error.WriteLine("не прочиталось: " + e.Message); return 1; }

var frames = reader.Frames;
if (frames.Count == 0) { Console.Error.WriteLine("в записи нет кадров"); return 1; }

// Тайминги кадров относительно первого, в миллисекундах (по часам источника).
ulong t0 = frames[0].CaptureNs;
var relMs = new double[frames.Count];
for (int i = 0; i < frames.Count; i++)
    relMs[i] = (frames[i].CaptureNs - t0) / 1e6;
double durMs = relMs[^1];
Console.WriteLine($"кадров: {frames.Count}, длительность ≈ {durMs / 1000:F1} с, " +
                  $"{frames[0].Width}×{frames[0].Height}");

var http = new HttpListener();
http.Prefixes.Add($"http://localhost:{port}/");
http.Start();
Console.WriteLine($"плеер: http://localhost:{port}/   (Ctrl+C для выхода)");

await Loop();
return 0;

async Task Loop()
{
    while (http.IsListening)
    {
        HttpListenerContext ctx;
        try { ctx = await http.GetContextAsync(); }
        catch { return; }
        _ = Task.Run(() => Handle(ctx));
    }
}

void Handle(HttpListenerContext ctx)
{
    try
    {
        string p = ctx.Request.Url?.AbsolutePath ?? "/";
        switch (p)
        {
            case "/":
                WriteText(ctx, "text/html; charset=utf-8", Html());
                break;

            case "/meta":
            {
                // JSON: число кадров, тайминги (мс), размеры, id кадров.
                var sb = new StringBuilder();
                sb.Append("{\"count\":").Append(frames.Count)
                  .Append(",\"w\":").Append(frames[0].Width)
                  .Append(",\"h\":").Append(frames[0].Height)
                  .Append(",\"durMs\":").Append(durMs.ToString("F3", CultureInfo.InvariantCulture))
                  .Append(",\"t\":[");
                for (int i = 0; i < relMs.Length; i++)
                {
                    if (i > 0) sb.Append(',');
                    sb.Append(relMs[i].ToString("F3", CultureInfo.InvariantCulture));
                }
                sb.Append("]}");
                WriteText(ctx, "application/json", sb.ToString());
                break;
            }

            case "/frame":
            {
                var q = ctx.Request.QueryString["i"];
                if (!int.TryParse(q, out int i) || i < 0 || i >= frames.Count)
                { ctx.Response.StatusCode = 400; ctx.Response.Close(); break; }
                var f = frames[i];
                var header = new byte[12];
                header[0] = (byte)(f.Width & 0xFF); header[1] = (byte)(f.Width >> 8);
                header[2] = (byte)(f.Height & 0xFF); header[3] = (byte)(f.Height >> 8);
                header[4] = (byte)f.PixFmt; header[5] = (byte)f.Compression;
                header[6] = (byte)(f.FrameId & 0xFF); header[7] = (byte)((f.FrameId >> 8) & 0xFF);
                header[8] = (byte)((f.FrameId >> 16) & 0xFF); header[9] = (byte)((f.FrameId >> 24) & 0xFF);
                ctx.Response.ContentType = "application/octet-stream";
                ctx.Response.Headers["Cache-Control"] = "no-store";
                ctx.Response.ContentLength64 = header.Length + f.Payload.Length;
                ctx.Response.OutputStream.Write(header, 0, header.Length);
                ctx.Response.OutputStream.Write(f.Payload, 0, f.Payload.Length);
                ctx.Response.OutputStream.Close();
                break;
            }

            default:
                ctx.Response.StatusCode = 404; ctx.Response.Close(); break;
        }
    }
    catch { try { ctx.Response.Abort(); } catch { } }
}

static void WriteText(HttpListenerContext ctx, string ct, string body)
{
    var b = Encoding.UTF8.GetBytes(body);
    ctx.Response.ContentType = ct;
    ctx.Response.ContentLength64 = b.Length;
    ctx.Response.OutputStream.Write(b, 0, b.Length);
    ctx.Response.OutputStream.Close();
}

static string Html() => """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MouthLink — плеер записи</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#111; color:#ddd; font:14px system-ui,sans-serif;
         display:flex; flex-direction:column; align-items:center; gap:10px; padding:16px; }
  canvas { image-rendering:pixelated; background:#000; max-width:95vw; border:1px solid #333; border-radius:6px; }
  #bar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:center; }
  button { background:#223; color:#cde; border:1px solid #345; border-radius:5px; padding:6px 10px; cursor:pointer; }
  button:hover { background:#2b3350; }
  #seek { width:min(80vw,520px); }
  #stat { font-variant-numeric:tabular-nums; color:#9bd; }
  select { background:#223; color:#cde; border:1px solid #345; border-radius:5px; padding:5px; }
</style>
</head>
<body>
  <div id="stat">загрузка…</div>
  <canvas id="cv" width="320" height="240"></canvas>
  <input id="seek" type="range" min="0" max="0" value="0" step="1">
  <div id="bar">
    <button id="first">⏮</button>
    <button id="prev">◀ шаг</button>
    <button id="play">▶ играть</button>
    <button id="next">шаг ▶</button>
    <button id="last">⏭</button>
    <label>скорость
      <select id="speed">
        <option value="0.25">0.25×</option>
        <option value="0.5">0.5×</option>
        <option value="1" selected>1×</option>
        <option value="2">2×</option>
        <option value="4">4×</option>
      </select>
    </label>
    <label><input type="checkbox" id="loop" checked> повтор</label>
    <label><input type="checkbox" id="fit" checked> ×2</label>
  </div>
<script>
const cv=document.getElementById('cv'), ctx=cv.getContext('2d'), stat=document.getElementById('stat');
const seek=document.getElementById('seek'), playBtn=document.getElementById('play');
const speedSel=document.getElementById('speed'), loopCb=document.getElementById('loop'), fitCb=document.getElementById('fit');
let meta=null, idx=0, playing=false, rafId=0, segStartPerf=0, segStartMs=0;

function scale(w,h){ const s=fitCb.checked?2:1; cv.style.width=(w*s)+'px'; cv.style.height=(h*s)+'px'; }

async function drawGray(buf,w,h,bpp){
  if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;scale(w,h);}
  const img=ctx.createImageData(w,h), out=img.data, n=w*h;
  if(bpp===1){ for(let i=0;i<n;i++){const v=buf[i],j=i<<2;out[j]=v;out[j+1]=v;out[j+2]=v;out[j+3]=255;} }
  else { let mx=1; for(let i=0;i<n;i++){const v=buf[i*2]|(buf[i*2+1]<<8);if(v>mx)mx=v;}
         for(let i=0;i<n;i++){const v=buf[i*2]|(buf[i*2+1]<<8);const g=(v*255/mx)|0,j=i<<2;out[j]=g;out[j+1]=g;out[j+2]=g;out[j+3]=255;} }
  ctx.putImageData(img,0,0);
}
async function drawJpeg(bytes,w,h){
  const bmp=await createImageBitmap(new Blob([bytes],{type:'image/jpeg'}));
  if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;scale(w,h);}
  ctx.drawImage(bmp,0,0,w,h); bmp.close();
}
async function show(i){
  const r=await fetch('/frame?i='+i,{cache:'no-store'}); if(!r.ok)return;
  const ab=await r.arrayBuffer(), dv=new DataView(ab);
  const w=dv.getUint16(0,true),h=dv.getUint16(2,true),pf=dv.getUint8(4),fid=dv.getUint32(6,true);
  const pay=new Uint8Array(ab,12);
  if(pf===0x10)await drawJpeg(pay,w,h); else if(pf===0x02)await drawGray(pay,w,h,2); else await drawGray(pay,w,h,1);
  const tms=meta.t[i];
  stat.textContent=`кадр ${i+1}/${meta.count}  #${fid}  ${(tms/1000).toFixed(2)} с  ${w}×${h}  fmt=0x${pf.toString(16)}`;
  seek.value=i;
}
function setIdx(i){ idx=Math.max(0,Math.min(meta.count-1,i)); show(idx); }

function startSeg(){ segStartPerf=performance.now(); segStartMs=meta.t[idx]; }
function frameAtMs(ms){ // первый кадр с t >= ms
  let lo=0,hi=meta.count-1,ans=meta.count-1;
  while(lo<=hi){const m=(lo+hi)>>1; if(meta.t[m]>=ms){ans=m;hi=m-1;}else lo=m+1;}
  return ans;
}
function loopPlay(){
  if(!playing)return;
  const sp=parseFloat(speedSel.value);
  const curMs=segStartMs+(performance.now()-segStartPerf)*sp;
  if(curMs>=meta.durMs){
    if(loopCb.checked){ setIdx(0); startSeg(); }
    else { pause(); setIdx(meta.count-1); return; }
  } else { const i=frameAtMs(curMs); if(i!==idx){idx=i;show(i);} }
  rafId=requestAnimationFrame(loopPlay);
}
function play(){ if(playing)return; if(idx>=meta.count-1)setIdx(0); playing=true; playBtn.textContent='⏸ пауза'; startSeg(); rafId=requestAnimationFrame(loopPlay); }
function pause(){ playing=false; playBtn.textContent='▶ играть'; cancelAnimationFrame(rafId); }

playBtn.onclick=()=>playing?pause():play();
document.getElementById('first').onclick=()=>{pause();setIdx(0);};
document.getElementById('last').onclick=()=>{pause();setIdx(meta.count-1);};
document.getElementById('prev').onclick=()=>{pause();setIdx(idx-1);};
document.getElementById('next').onclick=()=>{pause();setIdx(idx+1);};
seek.oninput=()=>{pause();setIdx(parseInt(seek.value));};
speedSel.onchange=()=>{ if(playing)startSeg(); };
fitCb.onchange=()=>scale(cv.width,cv.height);
document.addEventListener('keydown',e=>{
  if(e.code==='Space'){e.preventDefault();playing?pause():play();}
  else if(e.code==='ArrowLeft'){pause();setIdx(idx-1);}
  else if(e.code==='ArrowRight'){pause();setIdx(idx+1);}
});

(async()=>{
  meta=await (await fetch('/meta')).json();
  seek.max=meta.count-1;
  await setIdx(0);
  stat.textContent+=' — пробел: играть/пауза, ←/→: шаг';
})();
</script>
</body>
</html>
""";

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using MouthLink.Core;
using MouthLink.Receiver;

// Приёмник кадров на ПК. Подписывается у источника (эмулятора или шлема),
// собирает фрагменты в кадры, считает статистику, при желании пишет сессию.
//
// Запуск:
//   dotnet run --project MouthLink.Receiver -- --host 127.0.0.1 --ctrl 9040 --listen 9041 [--fps 60] [--record out.mlrec] [--roi x,y,w,h]
//
//   --fps 0       без ограничения (каждый кадр источника)
//   --no-stock    не просить у шлема стоковые значения рта (STOCK). По умолчанию просим;
//                 при --record они пишутся рядом: <имя>.stock.csv (нужен модуль ml3+).
//   --roi x,y,w,h источник присылает только этот прямоугольник кадра (экономит полосу).
//                 Если прямоугольник не помещается в кадр, источник шлёт полный кадр.
//   Ctrl+C        корректно дописывает запись и отписывается от источника.
//
//   Калибровка с подсказками (нужен --record):
//   --prompts tongue  голосом/в консоли говорит, что делать языком, пишет <имя>.prompts.csv;
//                     по окончании подсказок запись останавливается сама.
//   --prompts face    полный сценарий выражений из каталога prompts/face.json (≈7.5 мин на круг,
//                     время удержания у каждого выражения своё); --rounds по умолчанию 1
//   --skip id,id      пропустить выражения (например tongue_roll, если язык не сворачивается)
//   --groups g,g      только группы: jaw, lips, corners, nose, cheeks, tongue, tongue_jaw, emotion, speech
//   --catalog FILE    другой каталог выражений
//   --rounds 3 --hold 4 --rest 2.5   (tongue) кругов, секунд на движение, секунд «убрать»
//   --voice ru|en|off                 голос подсказок (ru — если в Windows есть русский голос)
//   --seed N                          порядок движений в кругах (по умолчанию случайный)
//
// Для локальной отладки сначала запусти MouthLink.Emulator, потом приёмник.

string host = "127.0.0.1";
int ctrlPort = 9040, listenPort = 9041, fps = 60, viewPort = 8080;
string? recordPath = null;
bool verify = false;   // --verify: сверять кадры с тестовым шаблоном (эмулятор --pattern)
ushort roiX = 0, roiY = 0, roiW = 0, roiH = 0;   // --roi x,y,w,h; w/h = 0 → полный кадр
bool wantStock = true;                              // --no-stock выключает
string? promptsProgram = null; int rounds = 3; double hold = 4, rest = 2.5; string voiceMode = "ru";
int seed = Environment.TickCount;
bool roundsSet = false; string? catalogPath = null; double timeScale = 1;
var skipIds = new HashSet<string>(); HashSet<string>? onlyGroups = null;

// Значение опции: следующий аргумент, если он есть и сам не опция.
string Val(int i) =>
    i + 1 < args.Length && !args[i + 1].StartsWith("--")
        ? args[i + 1]
        : throw new ArgumentException($"у {args[i]} нет значения");

try
{
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--host": host = Val(i++); break;
        case "--ctrl": ctrlPort = int.Parse(Val(i++)); break;
        case "--listen": listenPort = int.Parse(Val(i++)); break;
        case "--fps": fps = int.Parse(Val(i++)); break;
        case "--record": recordPath = Val(i++); break;
        case "--view": viewPort = int.Parse(Val(i++)); break; // 0 = выключить просмотр
        case "--verify": verify = true; break;
        case "--no-stock": wantStock = false; break;
        case "--prompts": promptsProgram = Val(i++); break;
        case "--rounds": rounds = int.Parse(Val(i++)); roundsSet = true; break;
        case "--skip": foreach (var x in Val(i++).Split(',', StringSplitOptions.RemoveEmptyEntries)) skipIds.Add(x.Trim()); break;
        case "--groups": onlyGroups = new HashSet<string>(Val(i++).Split(',', StringSplitOptions.RemoveEmptyEntries).Select(x => x.Trim())); break;
        case "--catalog": catalogPath = Val(i++); break;
        case "--time-scale": timeScale = double.Parse(Val(i++), System.Globalization.CultureInfo.InvariantCulture); break; // для тестов
        case "--hold": hold = double.Parse(Val(i++), System.Globalization.CultureInfo.InvariantCulture); break;
        case "--rest": rest = double.Parse(Val(i++), System.Globalization.CultureInfo.InvariantCulture); break;
        case "--voice": voiceMode = Val(i++); break;
        case "--seed": seed = int.Parse(Val(i++)); break;
        case "--roi":
        {
            var parts = Val(i++).Split(',');
            if (parts.Length != 4 ||
                !ushort.TryParse(parts[0], out roiX) || !ushort.TryParse(parts[1], out roiY) ||
                !ushort.TryParse(parts[2], out roiW) || !ushort.TryParse(parts[3], out roiH) ||
                roiW == 0 || roiH == 0 || roiX + roiW > 65535 || roiY + roiH > 65535)
            {
                Console.WriteLine("--roi ожидает x,y,w,h (целые, w и h > 0), например --roi 120,140,280,260");
                return 2;
            }
            break;
        }
        default:
            throw new ArgumentException($"неизвестная опция «{args[i]}»");
    }
}
}
catch (Exception e) when (e is ArgumentException or FormatException or OverflowException)
{
    Console.WriteLine(e is ArgumentException ? e.Message : $"неверное число в аргументах: {e.Message}");
    Console.WriteLine("пример: --host <IP-шлема> --fps 0 --record d:\\mouthrecon\\f01.mlrec --prompts face");
    return 2;
}
if (verify) Console.WriteLine("сверка с тестовым шаблоном включена (источник должен слать --pattern)");
if (promptsProgram != null && (promptsProgram is not ("tongue" or "face") || recordPath == null))
{
    Console.WriteLine(promptsProgram is not ("tongue" or "face") ? $"неизвестная программа подсказок «{promptsProgram}» (есть: tongue, face)"
                                                                 : "--prompts требует --record");
    return 2;
}
if (roiW > 0) Console.WriteLine($"ROI: x={roiX} y={roiY} {roiW}x{roiH} (если не влезает в кадр — источник шлёт полный кадр)");

var srcCtrl = new IPEndPoint(Dns.GetHostAddresses(host)[0], ctrlPort);
using var sock = new UdpClient(new IPEndPoint(IPAddress.Any, listenPort));
sock.Client.ReceiveBufferSize = 8 << 20; // крупный буфер ядра против всплесков потерь
// Таймаут приёма, чтобы цикл регулярно проверял флаг остановки (Ctrl+C), а не
// висел в Receive. Таймаут — обычный SocketException, цикл его пропускает.
sock.Client.ReceiveTimeout = 250;

// Windows: не падать от ICMP "port unreachable". Пока источник (эмулятор/шлем)
// ещё не слушает 9040, наши SUBSCRIBE/PING летят в пустоту, Windows шлёт ICMP
// "порт недоступен" и помечает сокет — тогда следующий Receive кидает
// WSAECONNRESET (10054). SIO_UDP_CONNRESET отключает это поведение.
if (OperatingSystem.IsWindows())
{
    try { sock.Client.IOControl(unchecked((int)0x9800000C), new byte[4], null); } catch { }
}

var clock = Stopwatch.StartNew();
ulong NowNs() => (ulong)(clock.Elapsed.TotalSeconds * 1e9);

var reasm = new FrameReassembler(maxPending: 4);
uint lastFrameId = 0; bool haveLastId = false;
long idGaps = 0, idOutOfOrder = 0;
long framesVerified = 0, verifyBadFrames = 0; int worstMismatch = 0;
SessionWriter? rec = recordPath != null ? new SessionWriter(recordPath) : null;
string? stockPath = rec != null && wantStock ? Path.ChangeExtension(recordPath!, ".stock.csv") : null;
StockCsvWriter? stockLog = stockPath != null ? new StockCsvWriter(stockPath) : null;
if (stockLog != null) Console.WriteLine($"стоковые значения рта → {stockPath}");
long stockCount = 0, stockBad = 0;
bool stockHintShown = false;
// Запись и закрытие файла — под одной блокировкой: закрытие из обработчика выхода
// не должно попасть в середину записи кадра (иначе в файле обрезанный хвост).
object recLock = new();
void CloseRecording()
{
    lock (recLock)
    {
        if (stockLog != null)
        {
            long rows = stockLog.Rows;
            stockLog.Dispose();
            stockLog = null;
            Console.WriteLine($"сток закрыт: {rows} строк → {stockPath}");
        }
        if (rec == null) return;
        long n = rec.FramesWritten;
        rec.Dispose();
        rec = null;
        Console.WriteLine($"запись закрыта: {n} кадров → {recordPath}");
    }
}
if (rec != null) Console.WriteLine($"запись сессии → {recordPath}");

LiveViewServer? live = null;
if (viewPort > 0)
{
    try { live = new LiveViewServer(viewPort); live.Start(); }
    catch (Exception e) { Console.WriteLine($"просмотр не поднялся на :{viewPort} ({e.Message}); продолжаю без него"); }
}

long bytesThisSec = 0;
long completedAtLastReport = 0;
double lastReportSec = 0;
double clockOffsetNs = 0;   // t_pc ≈ t_src + offset
bool haveOffset = false;
double lastLatencyMs = 0;

// Фоновый таймер: подписка и ping раз в полсекунды.
// Буферы заводим один раз вне цикла: stackalloc внутри бесконечного цикла
// не освобождается до выхода из метода и через часы переполнил бы стек.
uint pingSeq = 0;
var stop = new CancellationTokenSource();
var timerThread = new Thread(() =>
{
    var sub = new byte[16];
    var ping = new byte[16];
    var s = new ControlPackets.Subscribe
    {
        ReplyPort = (ushort)listenPort, TargetFps = (byte)Math.Clamp(fps, 0, 255),
        RoiX = roiX, RoiY = roiY, RoiW = roiW, RoiH = roiH,
        Flags = wantStock ? ControlPackets.SubscribeFlags.WantStock : (byte)0,
    };
    while (!stop.IsCancellationRequested)
    {
        int n = s.Write(sub);
        try { sock.Send(sub, n, srcCtrl); } catch { }

        int pn = ControlPackets.WritePing(ping, pingSeq++, NowNs());
        try { sock.Send(ping, pn, srcCtrl); } catch { }

        Thread.Sleep(500);
    }
});
timerThread.IsBackground = true;
timerThread.Start();

// Часы шлема для подсказок: последний кадр (capture_ns) и момент его прихода в часах ПК.
long lastCapNs = 0, lastArrNs = 0;
long SourceNow()
{
    long arr = Interlocked.Read(ref lastArrNs), cap = Interlocked.Read(ref lastCapNs);
    long now = (long)NowNs();
    if (arr > 0 && now - arr < 1_000_000_000L) return cap + (now - arr);
    if (haveOffset) return (long)(now - clockOffsetNs);
    return -1;
}
Prompter? prompter = null;
if (promptsProgram != null)
{
    List<Prompter.Step> steps;
    if (promptsProgram == "face")
    {
        var (catalog, catalogSource) = Prompter.LoadFaceCatalog(catalogPath);
        var unknown = skipIds.Where(id => catalog.Items.All(it => it.Id != id)).ToList();
        if (unknown.Count > 0) Console.WriteLine($"  ! в каталоге нет выражений: {string.Join(", ", unknown)}");
        steps = Prompter.Face(catalog, roundsSet ? rounds : 1, seed, skipIds, onlyGroups, timeScale);
        int n = steps.Count(st => st.Label.Length > 0 && st.Label != catalog.Rest.Id);
        Console.WriteLine($"каталог: {catalogSource}; выражений в сценарии: {n}");
        if (n == 0) { Console.WriteLine("после --skip/--groups не осталось ни одного выражения"); return 2; }
    }
    else steps = Prompter.Tongue(rounds, hold * timeScale, rest * timeScale, seed);
    double total = 0; foreach (var st in steps) total += st.Seconds;
    Console.WriteLine($"калибровка «{promptsProgram}»: ≈{total / 60:F1} мин");
    prompter = new Prompter(steps, SourceNow,
        () => { long arr = Interlocked.Read(ref lastArrNs); return arr > 0 && (long)NowNs() - arr < 300_000_000L; },
        Path.ChangeExtension(recordPath!, ".prompts.csv"), voiceMode, stop.Token);
    prompter.Start();
}

Console.WriteLine($"приёмник слушает :{listenPort}, источник {host}:{ctrlPort}. Ctrl+C для выхода.");
// Ctrl+C: не убивать процесс сразу, а дать циклу выйти и закрыть запись целиком.
Console.CancelKeyPress += (_, e) => { e.Cancel = true; stop.Cancel(); };
// Закрытие окна консоли и прочие выходы: дописать файл под блокировкой.
AppDomain.CurrentDomain.ProcessExit += (_, _) => { stop.Cancel(); CloseRecording(); live?.Dispose(); };

while (!stop.IsCancellationRequested)
{
    if (prompter != null && prompter.Finished)
    {
        Console.WriteLine("подсказки закончились — запись остановлена автоматически");
        break;
    }
    IPEndPoint from = new(IPAddress.Any, 0);
    byte[] data;
    try { data = sock.Receive(ref from); }
    catch (SocketException) { continue; }        // таймаут или транзиентная ошибка — не повод падать
    catch { break; }                              // сокет закрыт (выход) — завершаемся

    bytesThisSec += data.Length;

    if (data.Length >= 1 && (PacketType)data[0] == PacketType.Pong)
    {
        if (ControlPackets.TryParsePong(data, out _, out var tSend, out var tSrc))
        {
            ulong tRecv = NowNs();
            double rttMs = (tRecv - tSend) / 1e6;
            lastLatencyMs = rttMs / 2;
            // Оценка сдвига: t_pc(в середине RTT) ≈ t_src. offset = t_pc_mid - t_src.
            double tPcMid = tSend + (tRecv - tSend) / 2.0;
            clockOffsetNs = tPcMid - tSrc;
            haveOffset = true;
        }
        continue;
    }

    if (data.Length >= 1 && (PacketType)data[0] == PacketType.Stock)
    {
        if (StockSample.TryParse(data, out var st) && st != null)
        {
            stockCount++;
            lock (recLock) stockLog?.Write(st);
        }
        else stockBad++;
        continue;
    }

    var frame = reasm.Feed(data);
    if (frame != null)
    {
        lock (recLock) rec?.Write(frame);
        Interlocked.Exchange(ref lastCapNs, (long)frame.CaptureNs);
        Interlocked.Exchange(ref lastArrNs, (long)NowNs());
        live?.Publish(frame);
        // Задержка «захват → сейчас» в часах ПК, если знаем сдвиг. Сдвиг бывает и
        // отрицательным (часы источника впереди) — считаем в double со знаком.
        if (haveOffset)
            lastLatencyMs = ((double)NowNs() - ((double)frame.CaptureNs + clockOffsetNs)) / 1e6;

        // Пропуски номеров кадров (кадр потерян целиком или сборщик выкинул неполный).
        if (haveLastId)
        {
            uint step = frame.FrameId - lastFrameId;              // беззнаково: переживает переполнение
            if (step == 0 || step >= 0x8000_0000) idOutOfOrder++; // повтор или кадр из прошлого
            else idGaps += step - 1;
        }
        lastFrameId = frame.FrameId; haveLastId = true;

        if (verify)
        {
            framesVerified++;
            if (frame.PixFmt != PixelFormat.Gray8 || frame.Compression != Compression.None)
                verifyBadFrames++;
            else
            {
                int bad = TestPattern.CountMismatches(frame.Payload, frame.Width, frame.Height, frame.FrameId);
                if (bad != 0)
                {
                    verifyBadFrames++;
                    if (bad < 0 || bad > worstMismatch) worstMismatch = bad < 0 ? frame.Width * frame.Height : bad;
                    if (verifyBadFrames <= 5)
                        Console.WriteLine($"  НЕСОВПАДЕНИЕ: кадр #{frame.FrameId} {frame.Width}x{frame.Height}, " +
                                          (bad < 0 ? "неверный размер payload" : $"{bad} пикс. не совпало"));
                }
            }
        }
    }

    double nowSec = clock.Elapsed.TotalSeconds;
    if (nowSec - lastReportSec >= 1.0)
    {
        long completed = reasm.FramesCompleted;
        double fpsNow = (completed - completedAtLastReport) / (nowSec - lastReportSec);
        double mbit = bytesThisSec * 8 / 1e6 / (nowSec - lastReportSec);
        Console.WriteLine(
            $"fps={fpsNow,5:F1}  всего={completed,-7} потеряно={reasm.FramesDropped,-5} битых={reasm.FragmentsBad,-5} " +
            $"пропуск_id={idGaps,-5} {mbit,5:F1} Мбит/с  задержка≈{lastLatencyMs,4:F0} мс  запись={rec?.FramesWritten ?? 0}" +
            (wantStock ? $"  сток={stockCount}" + (stockBad > 0 ? $" (битых {stockBad})" : "") : "") +
            (idOutOfOrder > 0 ? $"  вне_порядка={idOutOfOrder}" : "") +
            (verify ? $"  сверка: {framesVerified - verifyBadFrames}/{framesVerified} ок" +
                      (verifyBadFrames > 0 ? $", худший кадр {worstMismatch} пикс." : "") : ""));
        if (wantStock && !stockHintShown && stockCount == 0 && completed > 270)
        {
            Console.WriteLine("  ! кадры идут, а стоковых значений нет: на шлеме модуль старше ml3 (или запусти с --no-stock)");
            stockHintShown = true;
        }
        completedAtLastReport = completed;
        bytesThisSec = 0;
        lastReportSec = nowSec;
    }
}

// Выход: отписаться, чтобы шлем сразу перестал слать (не ждать истечения подписки),
// и аккуратно закрыть запись.
try { sock.Send(new byte[] { (byte)PacketType.Unsubscribe, FrameHeader.CurrentVersion }, 2, srcCtrl); } catch { }
Console.WriteLine($"остановлено: кадров {reasm.FramesCompleted}, потеряно {reasm.FramesDropped}, пропуск_id {idGaps}" +
                  (wantStock ? $", сток {stockCount}" + (stockBad > 0 ? $" (битых {stockBad})" : "") : ""));
CloseRecording();
live?.Dispose();
return 0;

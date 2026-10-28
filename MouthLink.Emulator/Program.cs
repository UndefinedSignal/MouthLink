using System;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using MouthLink.Core;

// Эмулятор источника кадров. Стоит на месте шлема: слушает управляющий порт,
// по SUBSCRIBE шлёт синтетические GRAY8-кадры на указанный порт, отвечает на
// PING и DISCOVER. Нужен, чтобы отлаживать всю сторону ПК без шлема.
//
// Запуск:  dotnet run --project MouthLink.Emulator -- [ctrlPort] [width] [height] [fps] [--pattern]
// По умолчанию: 9040, 320, 240, 60.
// --pattern — вместо «рта» слать детерминированный тестовый шаблон (TestPattern):
//             приёмник с --verify сверяет каждый кадр попиксельно.

bool pattern = Array.IndexOf(args, "--pattern") >= 0;
var pos = Array.FindAll(args, s => !s.StartsWith("--"));
int ctrlPort = pos.Length > 0 ? int.Parse(pos[0]) : 9040;
int width = pos.Length > 1 ? int.Parse(pos[1]) : 320;
int height = pos.Length > 2 ? int.Parse(pos[2]) : 240;
int maxFps = pos.Length > 3 ? int.Parse(pos[3]) : 60;

using var ctrl = new UdpClient(new IPEndPoint(IPAddress.Any, ctrlPort));
// Windows: глушим WSAECONNRESET от ICMP "port unreachable" (см. приёмник).
if (OperatingSystem.IsWindows())
{
    try { ctrl.Client.IOControl(unchecked((int)0x9800000C), new byte[4], null); } catch { }
}
var clock = Stopwatch.StartNew();
ulong NowNs() => (ulong)(clock.Elapsed.TotalSeconds * 1e9);

IPEndPoint? subscriber = null;
int targetFps = maxFps;
long subscribedUntilMs = 0;
long NowMs() => clock.ElapsedMilliseconds;

// Приём управления в отдельном потоке.
var ctrlThread = new Thread(() =>
{
    while (true)
    {
        IPEndPoint from = new(IPAddress.Any, 0);
        byte[] data;
        try { data = ctrl.Receive(ref from); }
        catch (SocketException) { continue; }   // транзиентная ошибка — продолжаем слушать
        catch { return; }                        // сокет закрыт — выходим
        if (data.Length < 2) continue;

        switch ((PacketType)data[0])
        {
            case PacketType.Subscribe when ControlPackets.Subscribe.TryParse(data, out var sub):
                subscriber = new IPEndPoint(from.Address, sub.ReplyPort);
                targetFps = sub.TargetFps == 0 ? maxFps : Math.Min(sub.TargetFps, maxFps);
                subscribedUntilMs = NowMs() + 3000;
                Console.WriteLine($"SUBSCRIBE от {from.Address}:{sub.ReplyPort}, fps={targetFps}");
                break;

            case PacketType.Unsubscribe:
                Console.WriteLine("UNSUBSCRIBE");
                subscriber = null;
                break;

            case PacketType.Ping when ControlPackets.TryParsePing(data, out var seq, out var tpc):
            {
                Span<byte> pong = stackalloc byte[24];
                int n = ControlPackets.WritePong(pong, seq, tpc, NowNs());
                ctrl.Send(pong.Slice(0, n).ToArray(), n, from);
                break;
            }

            case PacketType.Discover:
            {
                Span<byte> hello = stackalloc byte[8];
                int n = ControlPackets.WriteHello(hello, camerasActive: true,
                    (ushort)width, (ushort)height, PixelFormat.Gray8);
                ctrl.Send(hello.Slice(0, n).ToArray(), n, from);
                Console.WriteLine($"DISCOVER от {from} → HELLO");
                break;
            }
        }
    }
});
ctrlThread.IsBackground = true;
ctrlThread.Start();

Console.WriteLine($"Эмулятор на :{ctrlPort}, кадр {width}x{height} GRAY8, до {maxFps} fps" +
                  (pattern ? ", ТЕСТОВЫЙ ШАБЛОН" : "") + ". Ctrl+C для выхода.");

using var tx = new UdpClient();
var frame = new byte[width * height];
uint frameId = 0;
var sw = Stopwatch.StartNew();

// База темпа: отсчёт ведём от начала стрима, а не от запуска программы.
// Иначе при подписке через несколько секунд эмулятор «догоняет» упущенное
// время без пауз и выстреливает пачку кадров вместо ровного fps.
double basisMs = 0;
uint basisFrame = 0;
bool streaming = false;

while (true)
{
    long nowMs = NowMs();
    var sub = subscriber;
    if (sub == null || nowMs > subscribedUntilMs)
    {
        if (sub != null && nowMs > subscribedUntilMs)
        {
            Console.WriteLine("подписка истекла, стоп");
            subscriber = null;
        }
        streaming = false;
        Thread.Sleep(5);
        continue;
    }

    if (!streaming)   // старт (или возобновление) стрима — сбрасываем базу темпа
    {
        streaming = true;
        basisMs = sw.Elapsed.TotalMilliseconds;
        basisFrame = frameId;
    }

    if (pattern) TestPattern.Fill(frame, width, height, frameId);
    else RenderSyntheticMouth(frame, width, height, frameId);

    var header = new FrameHeader
    {
        PixFmt = PixelFormat.Gray8,
        Compression = Compression.None,
        FrameId = frameId,
        CaptureNs = NowNs(),
        Width = (ushort)width,
        Height = (ushort)height,
        RoiX = 0,
        RoiY = 0,
    };
    var dest = sub;
    Fragmenter.Send(header, frame, dg => tx.Send(dg.ToArray(), dg.Length, dest));
    frameId++;

    // Выдержать темп относительно начала стрима.
    double periodMs = 1000.0 / targetFps;
    double targetMs = basisMs + (frameId - basisFrame) * periodMs;
    double aheadMs = targetMs - sw.Elapsed.TotalMilliseconds;
    if (aheadMs > 1) Thread.Sleep((int)aheadMs);
    if (frameId % (uint)Math.Max(1, targetFps) == 0)
        Console.WriteLine($"отправлено {frameId} кадров");
}

// Простой «рот»: тёмный фон, светлый горизонтальный эллипс, раскрывается и
// закрывается по синусоиде. Этого хватает, чтобы глазами видеть движение
// и проверить, что сборка и просмотр работают.
static void RenderSyntheticMouth(byte[] buf, int w, int h, uint t)
{
    double phase = (t % 120) / 120.0 * 2 * Math.PI;
    double openness = 0.5 * (1 - Math.Cos(phase)); // 0..1
    double cx = w / 2.0, cy = h / 2.0;
    double rx = w * 0.28;
    double ry = h * (0.04 + 0.20 * openness);

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            double nx = (x - cx) / rx;
            double ny = (y - cy) / ry;
            double d = nx * nx + ny * ny;
            byte v;
            if (d <= 1.0) v = (byte)(40 + 40 * openness);           // тёмная полость рта
            else v = (byte)(150 + 40 * Math.Sin(x * 0.05 + t * 0.1)); // «кожа» с лёгкой текстурой
            buf[y * w + x] = v;
        }
    }
}

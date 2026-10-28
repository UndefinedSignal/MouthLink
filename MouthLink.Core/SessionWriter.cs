using System;
using System.Buffers.Binary;
using System.IO;

namespace MouthLink.Core;

/// <summary>
/// Запись сессии на диск в простой контейнер .mlrec для офлайн-прогонов.
/// Формат:
///   магия "MLR1" (4 байта)
///   u16 version=1, u16 reserved
///   далее записи кадров, каждая:
///     u32 recordLen (всё, что после этого поля)
///     u32 frameId
///     u64 captureNs
///     u8  pixfmt
///     u8  compression
///     u16 width, u16 height, u16 roiX, u16 roiY
///     u32 payloadLen
///     payload[payloadLen]
/// Записи лежат в порядке завершения кадров. Кадры — снимки лица: файл только
/// локально, в репозиторий не коммитить.
/// </summary>
public sealed class SessionWriter : IDisposable
{
    private readonly BinaryWriter _w;
    public long FramesWritten { get; private set; }

    public SessionWriter(string path)
    {
        var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 20);
        _w = new BinaryWriter(fs);
        _w.Write(new byte[] { (byte)'M', (byte)'L', (byte)'R', (byte)'1' });
        Span<byte> hdr = stackalloc byte[4];
        BinaryPrimitives.WriteUInt16LittleEndian(hdr.Slice(0, 2), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(hdr.Slice(2, 2), 0);
        _w.Write(hdr);
    }

    public void Write(AssembledFrame f)
    {
        int recordLen = 4 + 8 + 1 + 1 + 2 + 2 + 2 + 2 + 4 + f.Payload.Length;
        Span<byte> meta = stackalloc byte[4 + recordLen - f.Payload.Length];
        BinaryPrimitives.WriteUInt32LittleEndian(meta.Slice(0, 4), (uint)recordLen);
        BinaryPrimitives.WriteUInt32LittleEndian(meta.Slice(4, 4), f.FrameId);
        BinaryPrimitives.WriteUInt64LittleEndian(meta.Slice(8, 8), f.CaptureNs);
        meta[16] = (byte)f.PixFmt;
        meta[17] = (byte)f.Compression;
        BinaryPrimitives.WriteUInt16LittleEndian(meta.Slice(18, 2), f.Width);
        BinaryPrimitives.WriteUInt16LittleEndian(meta.Slice(20, 2), f.Height);
        BinaryPrimitives.WriteUInt16LittleEndian(meta.Slice(22, 2), f.RoiX);
        BinaryPrimitives.WriteUInt16LittleEndian(meta.Slice(24, 2), f.RoiY);
        BinaryPrimitives.WriteUInt32LittleEndian(meta.Slice(26, 4), (uint)f.Payload.Length);
        _w.Write(meta);
        _w.Write(f.Payload);
        FramesWritten++;
    }

    public void Dispose() => _w.Dispose();
}

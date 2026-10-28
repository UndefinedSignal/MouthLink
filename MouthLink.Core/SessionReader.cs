using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;

namespace MouthLink.Core;

/// <summary>
/// Чтение записи сессии .mlrec (парная к SessionWriter). Чисто файловая операция:
/// загружает кадры с диска для офлайн-просмотра и прогонов. Формат — см.
/// SessionWriter.
/// </summary>
public sealed class SessionReader
{
    public IReadOnlyList<AssembledFrame> Frames { get; }

    /// <summary>Загрузить все кадры записи в память. Для больших записей см. замечание ниже.</summary>
    public SessionReader(string path)
    {
        var frames = new List<AssembledFrame>();
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);
        Span<byte> magic = stackalloc byte[8];
        if (!ReadExactly(fs, magic) || magic[0] != 'M' || magic[1] != 'L' || magic[2] != 'R' || magic[3] != '1')
            throw new InvalidDataException("не .mlrec (нет магии MLR1)");

        var lenBuf = new byte[4];
        while (true)
        {
            if (!ReadExactly(fs, lenBuf)) break;              // чистый конец файла
            int recLen = (int)BinaryPrimitives.ReadUInt32LittleEndian(lenBuf);
            if (recLen < 26 || recLen > 128 * 1024 * 1024)
                throw new InvalidDataException($"неправдоподобная длина записи: {recLen}");
            var rec = new byte[recLen];
            if (!ReadExactly(fs, rec)) break;                 // обрезанный хвост — отдаём что прочли

            uint frameId = BinaryPrimitives.ReadUInt32LittleEndian(rec.AsSpan(0, 4));
            ulong capNs = BinaryPrimitives.ReadUInt64LittleEndian(rec.AsSpan(4, 8));
            var pixfmt = (PixelFormat)rec[12];
            var comp = (Compression)rec[13];
            ushort w = BinaryPrimitives.ReadUInt16LittleEndian(rec.AsSpan(14, 2));
            ushort h = BinaryPrimitives.ReadUInt16LittleEndian(rec.AsSpan(16, 2));
            ushort rx = BinaryPrimitives.ReadUInt16LittleEndian(rec.AsSpan(18, 2));
            ushort ry = BinaryPrimitives.ReadUInt16LittleEndian(rec.AsSpan(20, 2));
            uint plen = BinaryPrimitives.ReadUInt32LittleEndian(rec.AsSpan(22, 4));
            if (26 + (long)plen > rec.Length)
                throw new InvalidDataException("длина payload выходит за запись");

            var payload = new byte[plen];
            Array.Copy(rec, 26, payload, 0, (int)plen);
            frames.Add(new AssembledFrame
            {
                FrameId = frameId, CaptureNs = capNs, PixFmt = pixfmt, Compression = comp,
                Width = w, Height = h, RoiX = rx, RoiY = ry, Payload = payload,
            });
        }
        Frames = frames;
    }

    private static bool ReadExactly(Stream s, Span<byte> dst)
    {
        int got = 0;
        while (got < dst.Length)
        {
            int n = s.Read(dst.Slice(got));
            if (n == 0) return false;
            got += n;
        }
        return true;
    }
}

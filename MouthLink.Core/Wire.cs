using System;
using System.Buffers.Binary;

namespace MouthLink.Core;

public enum PacketType : byte
{
    Fragment = 0x01,
    Subscribe = 0x02,
    Unsubscribe = 0x03,
    Ping = 0x04,
    Pong = 0x05,
    Discover = 0x06,
    Hello = 0x07,
    Stock = 0x08,   // стоковые значения рта для разметки кадра (см. StockPacket.cs)
}

public enum PixelFormat : byte
{
    Gray8 = 0x01,
    Gray16Le = 0x02,
    Jpeg = 0x10,
}

public enum Compression : byte
{
    None = 0,
    Lz4 = 1,
    Jpeg = 2,
}

/// <summary>
/// Шапка FRAGMENT-датаграммы (40 байт). Раскладка — см. PROTOCOL.md.
/// Всё little-endian.
/// </summary>
public struct FrameHeader
{
    public const int Size = 42;
    public const byte CurrentVersion = 1;

    public byte Version;
    public PixelFormat PixFmt;
    public Compression Compression;
    public uint FrameId;
    public ulong CaptureNs;
    public ushort Width;
    public ushort Height;
    public ushort RoiX;
    public ushort RoiY;
    public uint PayloadLen;
    public ushort FragIndex;
    public ushort FragCount;
    public uint FragOffset;   // u32: смещение может превышать 64 КБ при больших кадрах
    public ushort FragLen;

    /// <summary>Записать заголовок в начало dst и проставить CRC32. dst.Length должен быть >= Size.</summary>
    public void Write(Span<byte> dst)
    {
        if (dst.Length < Size)
            throw new ArgumentException($"нужно минимум {Size} байт", nameof(dst));

        dst[0] = (byte)PacketType.Fragment;
        dst[1] = Version;
        dst[2] = (byte)PixFmt;
        dst[3] = (byte)Compression;
        BinaryPrimitives.WriteUInt32LittleEndian(dst.Slice(4, 4), FrameId);
        BinaryPrimitives.WriteUInt64LittleEndian(dst.Slice(8, 8), CaptureNs);
        BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(16, 2), Width);
        BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(18, 2), Height);
        BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(20, 2), RoiX);
        BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(22, 2), RoiY);
        BinaryPrimitives.WriteUInt32LittleEndian(dst.Slice(24, 4), PayloadLen);
        BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(28, 2), FragIndex);
        BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(30, 2), FragCount);
        BinaryPrimitives.WriteUInt32LittleEndian(dst.Slice(32, 4), FragOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(36, 2), FragLen);
        uint crc = Crc32.Compute(dst.Slice(0, 38));
        BinaryPrimitives.WriteUInt32LittleEndian(dst.Slice(38, 4), crc);
    }

    /// <summary>
    /// Разобрать заголовок из начала src. Возвращает false, если короткий,
    /// не тот type/version или CRC не сошёлся.
    /// </summary>
    public static bool TryParse(ReadOnlySpan<byte> src, out FrameHeader h)
    {
        h = default;
        if (src.Length < Size) return false;
        if (src[0] != (byte)PacketType.Fragment) return false;

        uint want = BinaryPrimitives.ReadUInt32LittleEndian(src.Slice(38, 4));
        if (Crc32.Compute(src.Slice(0, 38)) != want) return false;

        h.Version = src[1];
        h.PixFmt = (PixelFormat)src[2];
        h.Compression = (Compression)src[3];
        h.FrameId = BinaryPrimitives.ReadUInt32LittleEndian(src.Slice(4, 4));
        h.CaptureNs = BinaryPrimitives.ReadUInt64LittleEndian(src.Slice(8, 8));
        h.Width = BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(16, 2));
        h.Height = BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(18, 2));
        h.RoiX = BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(20, 2));
        h.RoiY = BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(22, 2));
        h.PayloadLen = BinaryPrimitives.ReadUInt32LittleEndian(src.Slice(24, 4));
        h.FragIndex = BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(28, 2));
        h.FragCount = BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(30, 2));
        h.FragOffset = BinaryPrimitives.ReadUInt32LittleEndian(src.Slice(32, 4));
        h.FragLen = BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(36, 2));
        return true;
    }
}

/// <summary>Табличный CRC32 (IEEE 802.3, полином 0xEDB88320).</summary>
public static class Crc32
{
    private static readonly uint[] Table = BuildTable();

    private static uint[] BuildTable()
    {
        var t = new uint[256];
        for (uint i = 0; i < 256; i++)
        {
            uint c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }

    public static uint Compute(ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (byte b in data)
            crc = Table[(crc ^ b) & 0xFF] ^ (crc >> 8);
        return crc ^ 0xFFFFFFFF;
    }
}

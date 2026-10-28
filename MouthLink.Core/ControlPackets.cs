using System;
using System.Buffers.Binary;

namespace MouthLink.Core;

/// <summary>Разбор/сборка управляющих датаграмм (SUBSCRIBE/PING/DISCOVER и ответы).</summary>
public static class ControlPackets
{
    // ---- SUBSCRIBE (0x02) ----
    public struct Subscribe
    {
        public ushort ReplyPort;
        public byte TargetFps;      // 0 = максимум источника
        public PixelFormat PixFmtReq; // 0 = на выбор источника
        public ushort RoiX, RoiY, RoiW, RoiH; // RoiW/H = 0 → полный кадр
        public byte Flags;            // байт 14: бит 0 — прислать ещё и STOCK (SubscribeFlags)

        /// <summary>16 байт: 14 исходных + флаги + резерв. Старый источник читает первые 14.</summary>
        public int Write(Span<byte> dst)
        {
            dst[0] = (byte)PacketType.Subscribe;
            dst[1] = FrameHeader.CurrentVersion;
            BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(2, 2), ReplyPort);
            dst[4] = TargetFps;
            dst[5] = (byte)PixFmtReq;
            BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(6, 2), RoiX);
            BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(8, 2), RoiY);
            BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(10, 2), RoiW);
            BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(12, 2), RoiH);
            dst[14] = Flags;
            dst[15] = 0;
            return 16;
        }

        public static bool TryParse(ReadOnlySpan<byte> s, out Subscribe v)
        {
            v = default;
            if (s.Length < 14 || s[0] != (byte)PacketType.Subscribe) return false;
            v.ReplyPort = BinaryPrimitives.ReadUInt16LittleEndian(s.Slice(2, 2));
            v.TargetFps = s[4];
            v.PixFmtReq = (PixelFormat)s[5];
            v.RoiX = BinaryPrimitives.ReadUInt16LittleEndian(s.Slice(6, 2));
            v.RoiY = BinaryPrimitives.ReadUInt16LittleEndian(s.Slice(8, 2));
            v.RoiW = BinaryPrimitives.ReadUInt16LittleEndian(s.Slice(10, 2));
            v.RoiH = BinaryPrimitives.ReadUInt16LittleEndian(s.Slice(12, 2));
            v.Flags = s.Length >= 15 ? s[14] : (byte)0;
            return true;
        }
    }

    public static class SubscribeFlags
    {
        public const byte WantStock = 0x01;
    }

    /// <summary>Возможности источника из 9-го байта HELLO (старые источники шлют 8 байт → 0).</summary>
    public static class HelloCaps
    {
        public const byte Stock = 0x01;
        public static byte Read(ReadOnlySpan<byte> hello) =>
            hello.Length >= 9 && hello[0] == (byte)PacketType.Hello ? hello[8] : (byte)0;
    }

    // ---- PING (0x04) / PONG (0x05) ----
    public static int WritePing(Span<byte> dst, uint seq, ulong tPcSendNs)
    {
        dst[0] = (byte)PacketType.Ping;
        dst[1] = FrameHeader.CurrentVersion;
        dst[2] = 0; dst[3] = 0;
        BinaryPrimitives.WriteUInt32LittleEndian(dst.Slice(4, 4), seq);
        BinaryPrimitives.WriteUInt64LittleEndian(dst.Slice(8, 8), tPcSendNs);
        return 16;
    }

    public static bool TryParsePing(ReadOnlySpan<byte> s, out uint seq, out ulong tPcSendNs)
    {
        seq = 0; tPcSendNs = 0;
        if (s.Length < 16 || s[0] != (byte)PacketType.Ping) return false;
        seq = BinaryPrimitives.ReadUInt32LittleEndian(s.Slice(4, 4));
        tPcSendNs = BinaryPrimitives.ReadUInt64LittleEndian(s.Slice(8, 8));
        return true;
    }

    public static int WritePong(Span<byte> dst, uint seq, ulong tPcSendNs, ulong tSrcNs)
    {
        dst[0] = (byte)PacketType.Pong;
        dst[1] = FrameHeader.CurrentVersion;
        dst[2] = 0; dst[3] = 0;
        BinaryPrimitives.WriteUInt32LittleEndian(dst.Slice(4, 4), seq);
        BinaryPrimitives.WriteUInt64LittleEndian(dst.Slice(8, 8), tPcSendNs);
        BinaryPrimitives.WriteUInt64LittleEndian(dst.Slice(16, 8), tSrcNs);
        return 24;
    }

    public static bool TryParsePong(ReadOnlySpan<byte> s, out uint seq, out ulong tPcSendNs, out ulong tSrcNs)
    {
        seq = 0; tPcSendNs = 0; tSrcNs = 0;
        if (s.Length < 24 || s[0] != (byte)PacketType.Pong) return false;
        seq = BinaryPrimitives.ReadUInt32LittleEndian(s.Slice(4, 4));
        tPcSendNs = BinaryPrimitives.ReadUInt64LittleEndian(s.Slice(8, 8));
        tSrcNs = BinaryPrimitives.ReadUInt64LittleEndian(s.Slice(16, 8));
        return true;
    }

    // ---- DISCOVER (0x06) / HELLO (0x07) ----
    public static int WriteDiscover(Span<byte> dst)
    {
        dst[0] = (byte)PacketType.Discover;
        dst[1] = FrameHeader.CurrentVersion;
        return 2;
    }

    /// <summary>HELLO: версия, флаг «камеры активны», родные размеры/формат (0 если неизвестно).</summary>
    public static int WriteHello(Span<byte> dst, bool camerasActive, ushort nativeW, ushort nativeH, PixelFormat fmt)
    {
        dst[0] = (byte)PacketType.Hello;
        dst[1] = FrameHeader.CurrentVersion;
        dst[2] = (byte)(camerasActive ? 1 : 0);
        dst[3] = (byte)fmt;
        BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(4, 2), nativeW);
        BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(6, 2), nativeH);
        return 8;
    }

    public static bool TryParseHello(ReadOnlySpan<byte> s, out bool camerasActive, out ushort w, out ushort h, out PixelFormat fmt)
    {
        camerasActive = false; w = 0; h = 0; fmt = 0;
        if (s.Length < 8 || s[0] != (byte)PacketType.Hello) return false;
        camerasActive = s[2] != 0;
        fmt = (PixelFormat)s[3];
        w = BinaryPrimitives.ReadUInt16LittleEndian(s.Slice(4, 2));
        h = BinaryPrimitives.ReadUInt16LittleEndian(s.Slice(6, 2));
        return true;
    }
}

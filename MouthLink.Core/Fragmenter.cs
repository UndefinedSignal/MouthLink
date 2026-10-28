using System;

namespace MouthLink.Core;

/// <summary>
/// Режет готовый кадр (возможно уже сжатый) на FRAGMENT-датаграммы.
/// Вызывает callback на каждую датаграмму, чтобы не плодить аллокации.
/// </summary>
public static class Fragmenter
{
    /// <summary>Макс. данных кадра в одной датаграмме, чтобы с заголовком влезть в MTU 1500.</summary>
    public const int DefaultMaxFragLen = 1400;

    public delegate void SendDatagram(ReadOnlySpan<byte> datagram);

    public static void Send(
        in FrameHeader template,
        ReadOnlySpan<byte> payload,
        SendDatagram send,
        int maxFragLen = DefaultMaxFragLen)
    {
        if (maxFragLen < 1 || maxFragLen > 1400)
            throw new ArgumentOutOfRangeException(nameof(maxFragLen));

        int total = payload.Length;
        int fragCount = total == 0 ? 1 : (total + maxFragLen - 1) / maxFragLen;
        if (fragCount > ushort.MaxValue)
            throw new ArgumentException("кадр слишком велик для выбранного maxFragLen");

        Span<byte> dgram = stackalloc byte[FrameHeader.Size + maxFragLen];

        for (int i = 0; i < fragCount; i++)
        {
            int offset = i * maxFragLen;
            int len = Math.Min(maxFragLen, total - offset);
            if (len < 0) len = 0;

            var h = template;
            h.Version = FrameHeader.CurrentVersion;
            h.PayloadLen = (uint)total;
            h.FragIndex = (ushort)i;
            h.FragCount = (ushort)fragCount;
            h.FragOffset = (uint)offset;
            h.FragLen = (ushort)len;
            h.Write(dgram);

            if (len > 0)
                payload.Slice(offset, len).CopyTo(dgram.Slice(FrameHeader.Size, len));

            send(dgram.Slice(0, FrameHeader.Size + len));
        }
    }
}

using System;
using System.Collections.Generic;

namespace MouthLink.Core;

/// <summary>Собранный кадр, отдаётся потребителю.</summary>
public sealed class AssembledFrame
{
    public uint FrameId;
    public ulong CaptureNs;
    public PixelFormat PixFmt;
    public Compression Compression;
    public ushort Width;
    public ushort Height;
    public ushort RoiX;
    public ushort RoiY;
    public byte[] Payload = Array.Empty<byte>();   // всё ещё как на проводе (возможно сжато)
}

/// <summary>
/// Склеивает FRAGMENT-датаграммы в кадры. Не потокобезопасен: держи по одному
/// на поток приёма. Незавершённые кадры старше MaxPending выкидываются, чтобы
/// при потерях словарь не рос.
/// </summary>
public sealed class FrameReassembler
{
    private sealed class Partial
    {
        public byte[] Buffer = Array.Empty<byte>();
        public bool[] Got = Array.Empty<bool>();
        public int Remaining;
        public FrameHeader Head;
    }

    private readonly Dictionary<uint, Partial> _pending = new();
    private readonly int _maxPending;

    public long FramesCompleted { get; private set; }
    public long FramesDropped { get; private set; }     // выкинуто незавершёнными
    public long FragmentsBad { get; private set; }      // битый заголовок/несогласованность

    public FrameReassembler(int maxPending = 3)
    {
        _maxPending = Math.Max(1, maxPending);
    }

    /// <summary>
    /// Скормить одну датаграмму. Если ею кадр завершился — вернуть его, иначе null.
    /// </summary>
    public AssembledFrame? Feed(ReadOnlySpan<byte> datagram)
    {
        if (!FrameHeader.TryParse(datagram, out var h)) { FragmentsBad++; return null; }

        // Санитария: куски должны согласовываться и помещаться в кадр.
        if (h.FragCount == 0 || h.FragIndex >= h.FragCount) { FragmentsBad++; return null; }
        if (h.PayloadLen == 0 || h.PayloadLen > 64 * 1024 * 1024) { FragmentsBad++; return null; }
        if (datagram.Length < FrameHeader.Size + h.FragLen) { FragmentsBad++; return null; }
        if ((long)h.FragOffset + h.FragLen > h.PayloadLen) { FragmentsBad++; return null; }

        if (!_pending.TryGetValue(h.FrameId, out var p))
        {
            DropStale(h.FrameId);
            p = new Partial
            {
                Buffer = new byte[h.PayloadLen],
                Got = new bool[h.FragCount],
                Remaining = h.FragCount,
                Head = h,
            };
            _pending[h.FrameId] = p;
        }
        else if (p.Head.FragCount != h.FragCount || p.Head.PayloadLen != h.PayloadLen)
        {
            // Несогласованный фрагмент того же frame_id — не доверяем, начинаем заново.
            FragmentsBad++;
            _pending.Remove(h.FrameId);
            return null;
        }

        if (p.Got[h.FragIndex]) return null;  // дубликат

        datagram.Slice(FrameHeader.Size, h.FragLen).CopyTo(p.Buffer.AsSpan((int)h.FragOffset, h.FragLen));
        p.Got[h.FragIndex] = true;
        if (--p.Remaining > 0) return null;

        _pending.Remove(h.FrameId);
        FramesCompleted++;
        return new AssembledFrame
        {
            FrameId = h.FrameId,
            CaptureNs = h.CaptureNs,
            PixFmt = h.PixFmt,
            Compression = h.Compression,
            Width = h.Width,
            Height = h.Height,
            RoiX = h.RoiX,
            RoiY = h.RoiY,
            Payload = p.Buffer,
        };
    }

    // Выкинуть незавершённые кадры заметно старше текущего, плюс держать размер под контролем.
    private void DropStale(uint newestId)
    {
        if (_pending.Count < _maxPending) return;

        List<uint>? toRemove = null;
        foreach (var kv in _pending)
        {
            // беззнаковая разница: насколько этот кадр старше нового
            uint age = newestId - kv.Key;
            if (age < 0x8000_0000 && age >= (uint)_maxPending)
                (toRemove ??= new()).Add(kv.Key);
        }
        if (toRemove != null)
            foreach (var id in toRemove) { _pending.Remove(id); FramesDropped++; }

        // Жёсткий предел на случай хаоса в frame_id.
        while (_pending.Count >= _maxPending * 4)
        {
            uint oldest = 0; bool first = true;
            foreach (var id in _pending.Keys)
            {
                uint age = newestId - id;
                if (first || age > newestId - oldest) { oldest = id; first = false; }
            }
            _pending.Remove(oldest);
            FramesDropped++;
        }
    }
}

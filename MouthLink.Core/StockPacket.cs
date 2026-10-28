using System;
using System.Buffers.Binary;
using System.Globalization;
using System.IO;
using System.Text;

namespace MouthLink.Core;

/// <summary>
/// STOCK (0x08, версия 1, 424 байта): стоковые значения рта для одного кадра камеры.
/// Шлем (PicoET-Enhance-MouthLink ml3+) шлёт его на каждый новый принятый кадр рта,
/// если приёмник выставил SubscribeFlags.WantStock. SourceNs совпадает с CaptureNs
/// FRAGMENT того же кадра — спаривание по точному равенству.
///
///   0 type | 1 version | 2 flags (бит0 valid, бит1 warmup) | 3 int8 class (-1/0/1)
///   4 u32 seq | 8 u64 source_ns
///   16  f32 raw[35]      — выход нейросети стока (0..32 — каналы mouth33, 33 — вероятность качества)
///   156 f32 instant[33]  — после стоковой постобработки, БЕЗ временного сглаживания
///   288 f32 final[33]    — после сглаживания PicoET (то, что уходит в бридж/VRCFT)
///   420 u32 CRC32 по байтам 0..419
/// Векторы на 33 — в порядке mouth33 (см. MouthSlots / MouthNames).
/// </summary>
public sealed class StockSample
{
    public const int Size = 424;
    public const int RawCount = 35, MouthCount = 33;

    /// <summary>Индексы слотов PICO (0..51) для порядка mouth33 — таблица стока 0x360cac.</summary>
    public static readonly int[] MouthSlots =
    {
        43, 29, 37, 25, 32, 7, 15, 5, 49, 23, 27, 40, 10, 24, 50, 6, 20,
        22, 34, 48, 18, 14, 17, 9, 19, 21, 33, 42, 39, 8, 1, 13, 51,
    };

    /// <summary>Имена 52 слотов PICO (порядок PicoShape из UnifiedPicoModule).</summary>
    public static readonly string[] PicoShapeNames =
    {
        "EyeLookDownL", "NoseSneerL", "EyeLookInL", "BrowInnerUp", "BrowDownR", "MouthClose",
        "MouthLowerDownR", "JawOpen", "MouthUpperUpR", "MouthShrugUpper", "MouthFunnel",
        "EyeLookInR", "EyeLookDownR", "NoseSneerR", "MouthRollUpper", "JawRight", "BrowDownL",
        "MouthShrugLower", "MouthRollLower", "MouthSmileL", "MouthPressL", "MouthSmileR",
        "MouthPressR", "MouthDimpleR", "MouthLeft", "JawForward", "EyeSquintL", "MouthFrownL",
        "EyeBlinkL", "CheekSquintL", "BrowOuterUpL", "EyeLookUpL", "JawLeft", "MouthStretchL",
        "MouthPucker", "EyeLookUpR", "BrowOuterUpR", "CheekSquintR", "EyeBlinkR", "MouthUpperUpL",
        "MouthFrownR", "EyeSquintR", "MouthStretchR", "CheekPuff", "EyeLookOutL", "EyeLookOutR",
        "EyeWideR", "EyeWideL", "MouthRight", "MouthDimpleL", "MouthLowerDownL", "TongueOut",
    };

    public static readonly string[] MouthNames = Array.ConvertAll(MouthSlots, s => PicoShapeNames[s]);

    public uint Seq;
    public ulong SourceNs;
    public bool Valid, Warmup;
    public sbyte Classification;
    public readonly float[] Raw = new float[RawCount];
    public readonly float[] Instant = new float[MouthCount];
    public readonly float[] Final = new float[MouthCount];

    public static bool TryParse(ReadOnlySpan<byte> s, out StockSample? v)
    {
        v = null;
        if (s.Length != Size || s[0] != (byte)PacketType.Stock || s[1] != FrameHeader.CurrentVersion) return false;
        if (Crc32.Compute(s.Slice(0, 420)) != BinaryPrimitives.ReadUInt32LittleEndian(s.Slice(420, 4))) return false;
        var r = new StockSample
        {
            Valid = (s[2] & 1) != 0,
            Warmup = (s[2] & 2) != 0,
            Classification = unchecked((sbyte)s[3]),
            Seq = BinaryPrimitives.ReadUInt32LittleEndian(s.Slice(4, 4)),
            SourceNs = BinaryPrimitives.ReadUInt64LittleEndian(s.Slice(8, 8)),
        };
        for (int i = 0; i < RawCount; i++) r.Raw[i] = BinaryPrimitives.ReadSingleLittleEndian(s.Slice(16 + 4 * i, 4));
        for (int i = 0; i < MouthCount; i++)
        {
            r.Instant[i] = BinaryPrimitives.ReadSingleLittleEndian(s.Slice(156 + 4 * i, 4));
            r.Final[i] = BinaryPrimitives.ReadSingleLittleEndian(s.Slice(288 + 4 * i, 4));
        }
        v = r;
        return true;
    }

    /// <summary>Сборка пакета (для эмулятора и тестов).</summary>
    public int Write(Span<byte> dst)
    {
        dst.Slice(0, Size).Clear();
        dst[0] = (byte)PacketType.Stock;
        dst[1] = FrameHeader.CurrentVersion;
        dst[2] = (byte)((Valid ? 1 : 0) | (Warmup ? 2 : 0));
        dst[3] = unchecked((byte)Classification);
        BinaryPrimitives.WriteUInt32LittleEndian(dst.Slice(4, 4), Seq);
        BinaryPrimitives.WriteUInt64LittleEndian(dst.Slice(8, 8), SourceNs);
        for (int i = 0; i < RawCount; i++) BinaryPrimitives.WriteSingleLittleEndian(dst.Slice(16 + 4 * i, 4), Raw[i]);
        for (int i = 0; i < MouthCount; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(dst.Slice(156 + 4 * i, 4), Instant[i]);
            BinaryPrimitives.WriteSingleLittleEndian(dst.Slice(288 + 4 * i, 4), Final[i]);
        }
        BinaryPrimitives.WriteUInt32LittleEndian(dst.Slice(420, 4), Crc32.Compute(dst.Slice(0, 420)));
        return Size;
    }
}

/// <summary>
/// Журнал STOCK рядом с записью кадров: &lt;имя&gt;.stock.csv. Одна строка на пакет, в порядке
/// прихода. Колонки: source_ns, seq, valid, warmup, class, raw_0..raw_34,
/// inst_&lt;Имя&gt; ×33, fin_&lt;Имя&gt; ×33. Числа — инвариантная культура (точка).
/// </summary>
public sealed class StockCsvWriter : IDisposable
{
    private readonly StreamWriter _w;
    private readonly StringBuilder _sb = new(4096);
    public long Rows { get; private set; }

    public StockCsvWriter(string path)
    {
        _w = new StreamWriter(new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 16),
                              new UTF8Encoding(false));
        var h = new StringBuilder("source_ns,seq,valid,warmup,class");
        for (int i = 0; i < StockSample.RawCount; i++) h.Append(",raw_").Append(i);
        foreach (var n in StockSample.MouthNames) h.Append(",inst_").Append(n);
        foreach (var n in StockSample.MouthNames) h.Append(",fin_").Append(n);
        _w.WriteLine(h.ToString());
    }

    public void Write(StockSample s)
    {
        var ci = CultureInfo.InvariantCulture;
        _sb.Clear();
        _sb.Append(s.SourceNs.ToString(ci)).Append(',').Append(s.Seq.ToString(ci)).Append(',')
           .Append(s.Valid ? '1' : '0').Append(',').Append(s.Warmup ? '1' : '0').Append(',')
           .Append(s.Classification.ToString(ci));
        foreach (var x in s.Raw) _sb.Append(',').Append(x.ToString("R", ci));
        foreach (var x in s.Instant) _sb.Append(',').Append(x.ToString("R", ci));
        foreach (var x in s.Final) _sb.Append(',').Append(x.ToString("R", ci));
        _w.WriteLine(_sb.ToString());
        Rows++;
    }

    public void Dispose() => _w.Dispose();
}

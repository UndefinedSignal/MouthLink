using System;

namespace MouthLink.Core;

/// <summary>
/// Детерминированный тестовый шаблон GRAY8 для проверки передачи изображения.
/// Значение пикселя зависит от (x, y, frame_id), поэтому ловятся: сдвиг или
/// перепутанные фрагменты, неверный шаг строки, обрезка, подмена кадра другим
/// кадром, порча байтов. Та же формула реализована в tools/mlrec_check.py.
///
///   v(x, y, f) = (7x + 13y + 31f + ((x·y) >> 4)) mod 256
/// </summary>
public static class TestPattern
{
    public static byte Value(int x, int y, uint frameId) =>
        (byte)((7 * x + 13 * y + 31 * (int)(frameId & 0xFFFF) + ((x * y) >> 4)) & 0xFF);

    public static void Fill(Span<byte> dst, int width, int height, uint frameId)
    {
        for (int y = 0; y < height; y++)
        {
            var row = dst.Slice(y * width, width);
            for (int x = 0; x < width; x++)
                row[x] = Value(x, y, frameId);
        }
    }

    /// <summary>Сколько пикселей кадра не совпало с шаблоном (-1, если размер payload неверный).</summary>
    public static int CountMismatches(ReadOnlySpan<byte> frame, int width, int height, uint frameId)
    {
        if (frame.Length != width * height) return -1;
        int bad = 0;
        for (int y = 0; y < height; y++)
        {
            var row = frame.Slice(y * width, width);
            for (int x = 0; x < width; x++)
                if (row[x] != Value(x, y, frameId)) bad++;
        }
        return bad;
    }
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;

namespace MouthLink.Receiver;

/// <summary>
/// Запись с подсказками (калибровка). Голосом (Windows SAPI) и в консоли говорит, что
/// делать ртом, и пишет интервалы подсказок в часах ШЛЕМА в &lt;запись&gt;.prompts.csv:
///   start_ns,end_ns,label
/// Время шлема берётся от последнего пришедшего кадра (его capture_ns + сколько прошло
/// с его прихода), иначе — по сдвигу часов из PING/PONG. Ошибка — десятки мс, а
/// mlrec_pair всё равно отрезает начало каждой подсказки (время реакции).
/// </summary>
public sealed class Prompter
{
    public sealed record Step(string Label, string Ru, string En, double Seconds);

    // Метки, которые понимает tools/mlrec_pair.py --tongue.
    static readonly (string Label, string Ru, string En)[] TongueMoves =
    {
        ("tongue_left", "Язык влево", "Tongue left"),
        ("tongue_right", "Язык вправо", "Tongue right"),
        ("tongue_up", "Язык вверх", "Tongue up"),
        ("tongue_down", "Язык вниз", "Tongue down"),
        ("tongue_out", "Язык прямо", "Tongue straight out"),
    };

    public static List<Step> Tongue(int rounds, double hold, double rest, int seed)
    {
        var rng = new Random(seed);
        var steps = new List<Step>
        {
            new("", "Калибровка языка. Влево и вправо — в свою сторону. Начинаем.",
                "Tongue calibration. Left and right are your own sides. Starting.", 5),
            new("neutral", "Рот спокойно", "Relax your mouth", 3),
        };
        for (int r = 0; r < rounds; r++)
        {
            var order = (ValueTuple<string, string, string>[])TongueMoves.Clone();
            for (int i = order.Length - 1; i > 0; i--) { int j = rng.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
            foreach (var (label, ru, en) in order)
            {
                steps.Add(new(label, ru, en, hold));
                steps.Add(new("neutral", "Убрать", "Back", rest));
            }
        }
        steps.Add(new("", "Готово, спасибо", "Done, thank you", 2));
        return steps;
    }

    // ---- Программа "face": каталог выражений prompts/face.json ----
    public sealed class FaceCatalog
    {
        public Phrase Intro { get; set; } = new();
        public Phrase Rest { get; set; } = new();
        public Defaults Defaults { get; set; } = new();
        public List<Item> Items { get; set; } = new();
    }
    public sealed class Phrase { public string Id { get; set; } = "neutral"; public string Ru { get; set; } = ""; public string En { get; set; } = ""; public double Seconds { get; set; } }
    public sealed class Defaults { public double Hold { get; set; } = 4; }
    public sealed class Item
    {
        public string Id { get; set; } = "";
        public string Group { get; set; } = "";
        public string Ru { get; set; } = "";
        public string En { get; set; } = "";
        public double? Hold { get; set; }
        public bool Optional { get; set; }
    }

    /// <summary>Каталог: --catalog, иначе prompts/face.json в текущей папке, иначе встроенный.</summary>
    public static (FaceCatalog Catalog, string Source) LoadFaceCatalog(string? path)
    {
        var opts = new JsonSerializerOptions { PropertyNameCaseInsensitive = true, ReadCommentHandling = JsonCommentHandling.Skip };
        string? file = path ?? (File.Exists(Path.Combine("prompts", "face.json")) ? Path.Combine("prompts", "face.json") : null);
        if (file != null)
            return (JsonSerializer.Deserialize<FaceCatalog>(File.ReadAllText(file), opts)!, Path.GetFullPath(file));
        using var stream = typeof(Prompter).Assembly.GetManifestResourceStream("MouthLink.prompts.face.json")
            ?? throw new InvalidOperationException("встроенный каталог prompts/face.json не найден");
        return (JsonSerializer.Deserialize<FaceCatalog>(stream, opts)!, "встроенный prompts/face.json");
    }

    public static List<Step> Face(FaceCatalog c, int rounds, int seed, ISet<string> skip, ISet<string>? groups, double timeScale)
    {
        var rng = new Random(seed);
        var items = c.Items.Where(i => !skip.Contains(i.Id) && (groups == null || groups.Contains(i.Group))).ToList();
        var steps = new List<Step>
        {
            new("", c.Intro.Ru, c.Intro.En, c.Intro.Seconds * timeScale),
            new(c.Rest.Id, c.Rest.Ru, c.Rest.En, c.Rest.Seconds * timeScale),
        };
        for (int r = 0; r < rounds; r++)
        {
            var order = items.OrderBy(_ => rng.Next()).ToList();
            foreach (var i in order)
            {
                steps.Add(new(i.Id, i.Ru, i.En, (i.Hold ?? c.Defaults.Hold) * timeScale));
                steps.Add(new(c.Rest.Id, c.Rest.Ru, c.Rest.En, c.Rest.Seconds * timeScale));
            }
        }
        steps.Add(new("", "Готово, спасибо", "Done, thank you", 2 * timeScale));
        return steps;
    }

    readonly List<Step> _steps;
    readonly Func<long> _sourceNow;      // часы шлема, нс; -1 если неизвестно
    readonly Func<bool> _framesFlowing;  // кадры идут прямо сейчас
    readonly string _csvPath;
    readonly Speaker _voice;
    readonly CancellationToken _stop;
    public bool Finished { get; private set; }
    public int Written { get; private set; }

    public Prompter(List<Step> steps, Func<long> sourceNow, Func<bool> framesFlowing, string csvPath,
                    string voice, CancellationToken stop)
    {
        _steps = steps; _sourceNow = sourceNow; _framesFlowing = framesFlowing; _csvPath = csvPath; _stop = stop;
        _voice = new Speaker(voice);
    }

    public void Start()
    {
        var t = new Thread(Run) { IsBackground = true, Name = "prompter" };
        t.Start();
    }

    static void Banner(string text, string suffix = "")
    {
        Console.WriteLine();
        Console.WriteLine($"  ==========  {text.ToUpperInvariant()}  =========={suffix}");
        Console.WriteLine();
    }

    void Run()
    {
        using var csv = new StreamWriter(new FileStream(_csvPath, FileMode.Create, FileAccess.Write, FileShare.Read),
                                         new UTF8Encoding(false));
        csv.WriteLine("start_ns,end_ns,label");
        csv.Flush();
        Console.WriteLine($"подсказки → {_csvPath}; голос: {_voice.Description}");

        bool waitingShown = false;
        while (!_stop.IsCancellationRequested && !_framesFlowing())
        {
            if (!waitingShown) { Console.WriteLine("подсказки ждут кадры: надень шлем (и проверь, что бридж запущен)"); waitingShown = true; }
            Thread.Sleep(200);
        }

        foreach (var step in _steps)
        {
            if (_stop.IsCancellationRequested) break;
            long start = _sourceNow();
            Banner(step.Ru, step.Label.Length > 0 && step.Label != "neutral" ? $"   [{step.Label}, {step.Seconds:0.#} с]" : "");
            _voice.Say(step.Ru, step.En);
            var until = DateTime.UtcNow.AddSeconds(step.Seconds);
            while (DateTime.UtcNow < until && !_stop.IsCancellationRequested) Thread.Sleep(20);
            long end = _sourceNow();
            if (step.Label.Length == 0) continue;
            if (start < 0 || end <= start)
            {
                Console.WriteLine($"  (подсказка «{step.Label}» без времени шлема — кадров не было, пропускаю)");
                continue;
            }
            csv.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{start},{end},{step.Label}"));
            csv.Flush();
            Written++;
        }
        Finished = true;
        Console.WriteLine($"подсказки закончились: записано интервалов {Written}");
    }
}

/// <summary>Голос через Windows SAPI (COM, без пакетов). Нет русского голоса — английский; нет SAPI — сигнал.</summary>
sealed class Speaker
{
    readonly dynamic? _v;
    public bool Russian { get; }
    public string Description { get; } = "выключен";

    public Speaker(string mode)
    {
        if (mode == "off" || !OperatingSystem.IsWindows()) { Description = mode == "off" ? "выключен" : "нет (не Windows)"; return; }
        try
        {
            var type = Type.GetTypeFromProgID("SAPI.SpVoice");
            if (type == null) { Description = "нет SAPI — только сигнал"; return; }
            _v = Activator.CreateInstance(type);
            if (mode != "en")
            {
                try
                {
                    dynamic voices = _v!.GetVoices("Language=419", "");
                    if ((int)voices.Count > 0) { _v.Voice = voices.Item(0); Russian = true; }
                }
                catch { }
            }
            Description = Russian ? "русский (SAPI)" : "английский (SAPI; русского голоса нет — подсказки по-английски)";
        }
        catch (Exception e) { _v = null; Description = $"нет голоса ({e.GetType().Name}) — только сигнал"; }
    }

    public void Say(string ru, string en)
    {
        if (_v != null)
        {
            try { _v.Speak(Russian ? ru : en, 3); return; } // 1 = async, 2 = оборвать прошлую фразу
            catch { }
        }
        if (OperatingSystem.IsWindows()) { try { Console.Beep(880, 150); } catch { } }
    }
}

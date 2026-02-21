using System.Runtime.InteropServices;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

/// <summary>
/// Reads tensors from a safetensors file (the default weight format on Hugging Face Hub).
/// This replaces the Python "from transformers import GPT2LMHeadModel" pattern.
/// </summary>
public static class SafetensorsReader
{
    public static Dictionary<string, Tensor> ReadFile(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        // Safetensors layout: [8-byte header size][JSON header][tensor data]
        var headerSize = reader.ReadUInt64();
        var headerBytes = reader.ReadBytes((int)headerSize);
        var headerJson = System.Text.Encoding.UTF8.GetString(headerBytes);
        var header = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(headerJson)!;

        long dataStart = 8 + (long)headerSize;
        var result = new Dictionary<string, Tensor>();

        foreach (var (name, info) in header)
        {
            if (name == "__metadata__") continue;

            var dtype = info.GetProperty("dtype").GetString()!;
            var shape = info.GetProperty("shape").EnumerateArray()
                .Select(e => e.GetInt64()).ToArray();
            var offsets = info.GetProperty("data_offsets").EnumerateArray()
                .Select(e => e.GetInt64()).ToArray();

            long start = offsets[0];
            long end = offsets[1];
            int dataLength = (int)(end - start);

            stream.Seek(dataStart + start, SeekOrigin.Begin);
            var data = reader.ReadBytes(dataLength);

            var scalarType = dtype switch
            {
                "F32"  => ScalarType.Float32,
                "F16"  => ScalarType.Float16,
                "BF16" => ScalarType.BFloat16,
                "F64"  => ScalarType.Float64,
                "I32"  => ScalarType.Int32,
                "I64"  => ScalarType.Int64,
                "I16"  => ScalarType.Int16,
                "I8"   => ScalarType.Int8,
                "U8"   => ScalarType.Byte,
                "BOOL" => ScalarType.Bool,
                _ => throw new NotSupportedException($"Unsupported safetensors dtype: {dtype}")
            };

            var tensor = CreateTensor(data, shape, scalarType);
            result[name] = tensor;
        }

        return result;
    }

    private static Tensor CreateTensor(byte[] data, long[] shape, ScalarType scalarType)
    {
        var span = data.AsSpan();
        Tensor tensor = scalarType switch
        {
            ScalarType.Float32 => torch.tensor(MemoryMarshal.Cast<byte, float>(span).ToArray()),
            ScalarType.Float64 => torch.tensor(MemoryMarshal.Cast<byte, double>(span).ToArray()),
            ScalarType.Int32   => torch.tensor(MemoryMarshal.Cast<byte, int>(span).ToArray()),
            ScalarType.Int64   => torch.tensor(MemoryMarshal.Cast<byte, long>(span).ToArray()),
            ScalarType.Int16   => torch.tensor(MemoryMarshal.Cast<byte, short>(span).ToArray()),
            ScalarType.Float16 => torch.tensor(MemoryMarshal.Cast<byte, Half>(span).ToArray()).to(ScalarType.Float16),
            _ => throw new NotSupportedException($"Unsupported scalar type for tensor creation: {scalarType}")
        };
        return tensor.reshape(shape);
    }
}

namespace GPTSharpLib;

public static class TensorOperations
{
    private static Random random = new();

    public static ScalarType int64 => ScalarType.Int64;

    public static Tensor tensor<T>(T[] values, ScalarType dtype = ScalarType.Float32, object? device = null)
        where T : IConvertible => new(values.Select(value => Convert.ToSingle(value)).ToArray(), values.Length);

    public static Tensor tensor(float value, object? device = null) => new([value], 1);

    public static Tensor arange(long start, long end, ScalarType dtype = ScalarType.Int64, object? device = null)
    {
        if (end <= start || end - start > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(end));

        var values = new float[end - start];
        for (var index = 0; index < values.Length; index++)
            values[index] = start + index;
        return new Tensor(values, values.Length);
    }

    public static void manual_seed(int seed) => random = new Random(seed);

    public static IDisposable no_grad() => NoOpScope.Instance;

    public static IDisposable NewDisposeScope() => NoOpScope.Instance;

    public static Tensor tanh(Tensor input) => input.Tanh();

    public static Tensor pow(Tensor input, float exponent) => input.Pow(exponent);

    public static bool allclose(Tensor left, Tensor right, double rtol = 1e-5, double atol = 1e-8)
    {
        if (!left.shape.SequenceEqual(right.shape))
            return false;

        for (var index = 0; index < left.Length; index++)
        {
            var difference = MathF.Abs(left.values[index] - right.values[index]);
            if (difference > atol + rtol * MathF.Abs(right.values[index]))
                return false;
        }

        return true;
    }

    public static (Tensor values, Tensor indices) topk(Tensor input, int k, int dim = -1)
    {
        ValidateLastDimension(dim, input);
        if (k <= 0 || k > input.shape[^1])
            throw new ArgumentOutOfRangeException(nameof(k));

        var outputShape = input.shape.ToArray();
        outputShape[^1] = k;
        var values = new Tensor(outputShape);
        var indices = new Tensor(outputShape);
        var groupCount = input.Length / input.shape[^1];
        for (var group = 0; group < groupCount; group++)
        {
            var sourceOffset = group * input.shape[^1];
            var selected = Enumerable.Range(0, input.shape[^1])
                .OrderByDescending(index => input.values[sourceOffset + index])
                .Take(k)
                .ToArray();
            for (var index = 0; index < k; index++)
            {
                values.values[group * k + index] = input.values[sourceOffset + selected[index]];
                indices.values[group * k + index] = selected[index];
            }
        }

        return (values, indices);
    }

    public static (Tensor values, Tensor indices) sort(Tensor input, int dim = -1, bool descending = false)
    {
        ValidateLastDimension(dim, input);
        var values = new Tensor(input.shape);
        var indices = new Tensor(input.shape);
        var width = input.shape[^1];
        var groupCount = input.Length / width;
        for (var group = 0; group < groupCount; group++)
        {
            var sourceOffset = group * width;
            IEnumerable<int> ordered = Enumerable.Range(0, width)
                .OrderBy(index => input.values[sourceOffset + index]);
            if (descending)
                ordered = ordered.Reverse();

            var selected = ordered.ToArray();
            for (var index = 0; index < width; index++)
            {
                values.values[sourceOffset + index] = input.values[sourceOffset + selected[index]];
                indices.values[sourceOffset + index] = selected[index];
            }
        }

        return (values, indices);
    }

    public static Tensor cumsum(Tensor input, int dim = -1)
    {
        ValidateLastDimension(dim, input);
        var result = new Tensor(input.shape);
        var width = input.shape[^1];
        for (var offset = 0; offset < input.Length; offset += width)
        {
            var sum = 0f;
            for (var index = 0; index < width; index++)
            {
                sum += input.values[offset + index];
                result.values[offset + index] = sum;
            }
        }

        return result;
    }

    public static Tensor where(Tensor condition, Tensor whenTrue, Tensor whenFalse)
    {
        var selected = whenTrue.Clone();
        for (var index = 0; index < selected.Length; index++)
            selected.values[index] = condition.values[condition.Length == 1 ? 0 : index] != 0f
                ? whenTrue.values[whenTrue.Length == 1 ? 0 : index]
                : whenFalse.values[whenFalse.Length == 1 ? 0 : index];
        return selected;
    }

    public static Tensor multinomial(Tensor probabilities, int num_samples)
    {
        if (probabilities.Rank != 2 || num_samples != 1)
            throw new NotSupportedException("Only rank-2 multinomial sampling with one sample is supported.");

        var result = new Tensor(probabilities.shape[0], 1);
        for (var row = 0; row < probabilities.shape[0]; row++)
        {
            var threshold = (float)random.NextDouble();
            var cumulative = 0f;
            var selected = probabilities.shape[1] - 1;
            for (var column = 0; column < probabilities.shape[1]; column++)
            {
                cumulative += probabilities[row, column];
                if (threshold <= cumulative)
                {
                    selected = column;
                    break;
                }
            }
            result[row, 0] = selected;
        }

        return result;
    }

    public static Tensor gather(Tensor input, int dim, Tensor index)
    {
        ValidateLastDimension(dim, input);
        if (input.Rank != 2 || index.Rank != 2 || input.shape[0] != index.shape[0])
            throw new ArgumentException("Gather requires rank-2 input and index tensors with matching row counts.");

        var result = new Tensor(index.shape);
        for (var row = 0; row < index.shape[0]; row++)
        for (var column = 0; column < index.shape[1]; column++)
        {
            var sourceColumn = checked((int)index[row, column]);
            result[row, column] = input[row, sourceColumn];
        }

        return result;
    }

    public static Tensor cat(IEnumerable<Tensor> tensors, int dim)
    {
        var items = tensors.ToArray();
        if (items.Length == 0)
            throw new ArgumentException("At least one tensor is required.", nameof(tensors));
        if (items.Any(item => item.Rank != items[0].Rank))
            throw new ArgumentException("Tensor ranks must match.", nameof(tensors));

        dim = dim < 0 ? dim + items[0].Rank : dim;
        var outputShape = items[0].shape.ToArray();
        outputShape[dim] = items.Sum(item => item.shape[dim]);
        foreach (var item in items)
        for (var axis = 0; axis < item.Rank; axis++)
            if (axis != dim && item.shape[axis] != outputShape[axis])
                throw new ArgumentException("Tensor shapes must match outside the concatenation dimension.", nameof(tensors));

        if (dim != items[0].Rank - 1)
            throw new NotSupportedException("Only concatenation along the final dimension is supported.");

        var output = new Tensor(outputShape);
        var groupCount = output.Length / outputShape[^1];
        for (var group = 0; group < groupCount; group++)
        {
            var outputOffset = group * outputShape[^1];
            foreach (var item in items)
            {
                var width = item.shape[^1];
                Array.Copy(item.values, group * width, output.values, outputOffset, width);
                outputOffset += width;
            }
        }

        return output;
    }

    public static class nn
    {
        public static class functional
        {
            public static Tensor softmax(Tensor input, int dim = -1)
            {
                ValidateLastDimension(dim, input);
                var output = new Tensor(input.shape);
                var width = input.shape[^1];
                for (var offset = 0; offset < input.Length; offset += width)
                {
                    var maximum = input.values.AsSpan(offset, width).ToArray().Max();
                    var sum = 0f;
                    for (var index = 0; index < width; index++)
                    {
                        var value = MathF.Exp(input.values[offset + index] - maximum);
                        output.values[offset + index] = value;
                        sum += value;
                    }
                    for (var index = 0; index < width; index++)
                        output.values[offset + index] /= sum;
                }
                return output;
            }

            public static Tensor scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, bool is_causal = false)
            {
                if (query.Rank != 4 || key.Rank != 4 || value.Rank != 4 || !query.shape.SequenceEqual(key.shape) || !key.shape.SequenceEqual(value.shape))
                    throw new ArgumentException("Attention requires matching rank-4 query, key, and value tensors.");

                var (batches, heads, tokens, channels) = (query.shape[0], query.shape[1], query.shape[2], query.shape[3]);
                var output = new Tensor(query.shape);
                var scale = 1f / MathF.Sqrt(channels);
                for (var batch = 0; batch < batches; batch++)
                for (var head = 0; head < heads; head++)
                for (var target = 0; target < tokens; target++)
                {
                    var scores = new float[tokens];
                    var maximum = float.NegativeInfinity;
                    for (var source = 0; source < tokens; source++)
                    {
                        if (is_causal && source > target)
                        {
                            scores[source] = float.NegativeInfinity;
                            continue;
                        }
                        var score = 0f;
                        for (var channel = 0; channel < channels; channel++)
                            score += query[batch, head, target, channel] * key[batch, head, source, channel];
                        scores[source] = score * scale;
                        maximum = MathF.Max(maximum, scores[source]);
                    }
                    var sum = 0f;
                    for (var source = 0; source < tokens; source++)
                    {
                        scores[source] = float.IsNegativeInfinity(scores[source]) ? 0f : MathF.Exp(scores[source] - maximum);
                        sum += scores[source];
                    }
                    for (var source = 0; source < tokens; source++)
                    for (var channel = 0; channel < channels; channel++)
                        output[batch, head, target, channel] += scores[source] / sum * value[batch, head, source, channel];
                }
                return output;
            }
        }
    }

    private static void ValidateLastDimension(int dimension, Tensor input)
    {
        if (dimension != -1 && dimension != input.Rank - 1)
            throw new NotSupportedException("Only operations along the final dimension are supported.");
    }

    private sealed class NoOpScope : IDisposable
    {
        public static NoOpScope Instance { get; } = new();

        public void Dispose()
        {
        }
    }
}

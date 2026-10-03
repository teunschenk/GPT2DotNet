public sealed class LayerNorm : Module<Tensor, Tensor>
{
    private readonly int normalizedShape;
    private readonly float epsilon;

    public LayerNorm(int normalizedShape, float epsilon = 1e-5f)
        : base(nameof(LayerNorm))
    {
        if (normalizedShape <= 0)
            throw new ArgumentOutOfRangeException(nameof(normalizedShape));
        if (epsilon <= 0f || !float.IsFinite(epsilon))
            throw new ArgumentOutOfRangeException(nameof(epsilon));

        this.normalizedShape = normalizedShape;
        this.epsilon = epsilon;
        weight = Tensor.Ones(normalizedShape);
        bias = Tensor.Zeros(normalizedShape);

        RegisterParameter(nameof(weight), weight);
        RegisterParameter(nameof(bias), bias);
    }

    public Tensor weight { get; }

    public Tensor bias { get; }

    public override Tensor forward(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Rank == 0 || input.shape[^1] != normalizedShape)
            throw new ArgumentException("The tensor's final dimension must match the normalized shape.", nameof(input));

        var output = new Tensor(input.shape);
        var groupCount = input.Length / normalizedShape;
        for (var group = 0; group < groupCount; group++)
        {
            var offset = group * normalizedShape;
            var mean = 0f;
            for (var index = 0; index < normalizedShape; index++)
                mean += input.values[offset + index];
            mean /= normalizedShape;

            var variance = 0f;
            for (var index = 0; index < normalizedShape; index++)
            {
                var difference = input.values[offset + index] - mean;
                variance += difference * difference;
            }
            variance /= normalizedShape;

            var inverseStandardDeviation = 1f / MathF.Sqrt(variance + epsilon);
            for (var index = 0; index < normalizedShape; index++)
                output.values[offset + index] = (input.values[offset + index] - mean) * inverseStandardDeviation * weight.values[index] + bias.values[index];
        }

        return output;
    }
}

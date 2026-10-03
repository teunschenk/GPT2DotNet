public sealed class Linear : Module<Tensor, Tensor>
{
    private readonly int inFeatures;
    private readonly int outFeatures;

    public Linear(int inFeatures, int outFeatures, bool hasBias = true)
        : base(nameof(Linear))
    {
        if (inFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(inFeatures));
        if (outFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(outFeatures));

        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        weight = CreateWeights(outFeatures, inFeatures);
        bias = hasBias ? Tensor.Zeros(outFeatures) : null;

        RegisterParameter(nameof(weight), weight);
        if (bias is not null)
            RegisterParameter(nameof(bias), bias);
    }

    public Tensor weight { get; }

    public Tensor? bias { get; }

    public override Tensor forward(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Rank == 0 || input.shape[^1] != inFeatures)
            throw new ArgumentException("The tensor's final dimension must match the input feature count.", nameof(input));

        var outputShape = input.shape.ToArray();
        outputShape[^1] = outFeatures;
        var output = new Tensor(outputShape);
        var groupCount = input.Length / inFeatures;

        for (var group = 0; group < groupCount; group++)
        {
            var inputOffset = group * inFeatures;
            var outputOffset = group * outFeatures;
            for (var outputFeature = 0; outputFeature < outFeatures; outputFeature++)
            {
                var weightOffset = outputFeature * inFeatures;
                var value = bias?.values[outputFeature] ?? 0f;
                for (var inputFeature = 0; inputFeature < inFeatures; inputFeature++)
                    value += input.values[inputOffset + inputFeature] * weight.values[weightOffset + inputFeature];

                output.values[outputOffset + outputFeature] = value;
            }
        }

        return output;
    }

    private static Tensor CreateWeights(int outFeatures, int inFeatures)
    {
        var values = new float[checked(outFeatures * inFeatures)];
        for (var index = 0; index < values.Length; index += 2)
        {
            var first = 1f - Random.Shared.NextSingle();
            var second = Random.Shared.NextSingle();
            var magnitude = 0.02f * MathF.Sqrt(-2f * MathF.Log(first));

            values[index] = magnitude * MathF.Cos(2f * MathF.PI * second);
            if (index + 1 < values.Length)
                values[index + 1] = magnitude * MathF.Sin(2f * MathF.PI * second);
        }

        return new Tensor(values, outFeatures, inFeatures);
    }
}

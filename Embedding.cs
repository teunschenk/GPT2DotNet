public sealed class Embedding : IParameterModule
{
    private readonly int numEmbeddings;
    private readonly int embeddingDimension;

    public Embedding(int numEmbeddings, int embeddingDimension)
    {
        if (numEmbeddings <= 0)
            throw new ArgumentOutOfRangeException(nameof(numEmbeddings));
        if (embeddingDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(embeddingDimension));

        this.numEmbeddings = numEmbeddings;
        this.embeddingDimension = embeddingDimension;
        weight = CreateWeights(numEmbeddings, embeddingDimension);
    }

    public Tensor weight { get; set; }

    public IEnumerable<NamedParameter> named_parameters()
    {
        yield return new NamedParameter(nameof(weight), weight);
    }

    public Tensor forward(Tensor indices)
    {
        ArgumentNullException.ThrowIfNull(indices);
        ValidateWeight();

        var outputShape = new int[indices.Rank + 1];
        Array.Copy(indices.shape, outputShape, indices.Rank);
        outputShape[^1] = embeddingDimension;

        var output = new Tensor(outputShape);
        for (var index = 0; index < indices.Length; index++)
        {
            var token = indices.values[index];
            if (!float.IsFinite(token) || token != MathF.Truncate(token) || token < 0 || token >= numEmbeddings)
                throw new ArgumentOutOfRangeException(nameof(indices), "Embedding indices must be finite integer values within the vocabulary range.");

            Array.Copy(weight.values, (int)token * embeddingDimension, output.values, index * embeddingDimension, embeddingDimension);
        }

        return output;
    }

    private void ValidateWeight()
    {
        ArgumentNullException.ThrowIfNull(weight);
        if (weight.Rank != 2 || weight.shape[0] != numEmbeddings || weight.shape[1] != embeddingDimension)
            throw new InvalidOperationException("The embedding weight tensor must have shape [numEmbeddings, embeddingDimension].");
    }

    private static Tensor CreateWeights(int numEmbeddings, int embeddingDimension)
    {
        var values = new float[checked(numEmbeddings * embeddingDimension)];
        for (var index = 0; index < values.Length; index += 2)
        {
            var first = 1f - Random.Shared.NextSingle();
            var second = Random.Shared.NextSingle();
            var magnitude = 0.02f * MathF.Sqrt(-2f * MathF.Log(first));

            values[index] = magnitude * MathF.Cos(2f * MathF.PI * second);
            if (index + 1 < values.Length)
                values[index + 1] = magnitude * MathF.Sin(2f * MathF.PI * second);
        }

        return new Tensor(values, numEmbeddings, embeddingDimension);
    }
}

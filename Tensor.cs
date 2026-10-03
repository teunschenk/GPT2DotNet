public sealed class Tensor
{
    public readonly float[] values;
    public readonly int[] shape;
    public readonly int[] strides;

    public Tensor(params int[] shape)
        : this(new float[GetElementCount(shape)], shape, takeOwnership: true)
    {
    }

    public Tensor(float[] values, params int[] shape)
        : this(values, shape, takeOwnership: false)
    {
    }

    private Tensor(float[] values, int[] shape, bool takeOwnership)
    {
        ArgumentNullException.ThrowIfNull(values);
        ValidateShape(shape);

        var elementCount = GetElementCount(shape);
        if (values.Length != elementCount)
            throw new ArgumentException("The number of values must match the tensor shape.", nameof(values));

        this.values = takeOwnership ? values : values.ToArray();
        this.shape = shape.ToArray();
        strides = CreateStrides(this.shape);
    }

    public int Rank => shape.Length;

    public int Length => values.Length;

    public IReadOnlyList<int> Shape => shape;

    public long size(int dimension)
    {
        if (dimension < 0)
            dimension += Rank;
        if ((uint)dimension >= (uint)Rank)
            throw new ArgumentOutOfRangeException(nameof(dimension));

        return shape[dimension];
    }

    public float this[params int[] indices]
    {
        get => values[GetFlatIndex(indices)];
        set => values[GetFlatIndex(indices)] = value;
    }

    public static Tensor Zeros(params int[] shape) => new(shape);

    public static Tensor Ones(params int[] shape)
    {
        var tensor = new Tensor(shape);
        Array.Fill(tensor.values, 1f);
        return tensor;
    }

    public static Tensor FromArray(float[] values, params int[] shape) => new(values, shape);

    public Tensor Clone() => new(values, shape);

    public Tensor Reshape(params int[] newShape)
    {
        if (GetElementCount(newShape) != Length)
            throw new ArgumentException("The new shape must contain the same number of elements.", nameof(newShape));

        return new Tensor(values, newShape);
    }

    public Tensor view(params int[] newShape) => Reshape(newShape);

    public Tensor contiguous() => Clone();

    public Tensor transpose(int firstDimension, int secondDimension)
    {
        firstDimension = NormalizeDimension(firstDimension);
        secondDimension = NormalizeDimension(secondDimension);
        if (firstDimension == secondDimension)
            return Clone();

        var dimensions = Enumerable.Range(0, Rank).ToArray();
        (dimensions[firstDimension], dimensions[secondDimension]) = (dimensions[secondDimension], dimensions[firstDimension]);
        return Transpose(dimensions);
    }

    public Tensor t()
    {
        if (Rank != 2)
            throw new InvalidOperationException("Transpose shorthand requires a rank-2 tensor.");

        return transpose(0, 1);
    }

    public Tensor[] split(int splitSize, int dim = 0)
    {
        if (splitSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(splitSize));

        dim = NormalizeDimension(dim);
        if (shape[dim] % splitSize != 0)
            throw new ArgumentException("The dimension size must be divisible by the split size.", nameof(splitSize));

        var result = new Tensor[shape[dim] / splitSize];
        for (var part = 0; part < result.Length; part++)
            result[part] = Slice(dim, part * splitSize, splitSize);

        return result;
    }

    public Tensor Slice(int dimension, int start, int length)
    {
        dimension = NormalizeDimension(dimension);
        if (start < 0 || length <= 0 || start > shape[dimension] - length)
            throw new ArgumentOutOfRangeException(nameof(start));

        var outputShape = shape.ToArray();
        outputShape[dimension] = length;
        var output = new Tensor(outputShape);
        var outputIndices = new int[Rank];
        var sourceIndices = new int[Rank];

        for (var flatIndex = 0; flatIndex < output.Length; flatIndex++)
        {
            GetIndices(flatIndex, output.strides, outputIndices);
            outputIndices.CopyTo(sourceIndices, 0);
            sourceIndices[dimension] += start;
            output.values[flatIndex] = values[GetFlatIndex(sourceIndices)];
        }

        return output;
    }

    public Tensor unsqueeze(int dimension)
    {
        if (dimension < 0)
            dimension += Rank + 1;
        if ((uint)dimension > (uint)Rank)
            throw new ArgumentOutOfRangeException(nameof(dimension));

        var newShape = new int[Rank + 1];
        Array.Copy(shape, 0, newShape, 0, dimension);
        newShape[dimension] = 1;
        Array.Copy(shape, dimension, newShape, dimension + 1, Rank - dimension);
        return new Tensor(values, newShape);
    }

    public Tensor repeat(params int[] repeats)
    {
        if (repeats.Length != Rank || repeats.Any(repeat => repeat <= 0))
            throw new ArgumentException("A positive repeat count is required for every tensor dimension.", nameof(repeats));

        var outputShape = shape.Zip(repeats, (dimension, repeat) => checked(dimension * repeat)).ToArray();
        var output = new Tensor(outputShape);
        var outputIndices = new int[Rank];
        var sourceIndices = new int[Rank];
        for (var flatIndex = 0; flatIndex < output.Length; flatIndex++)
        {
            GetIndices(flatIndex, output.strides, outputIndices);
            for (var axis = 0; axis < Rank; axis++)
                sourceIndices[axis] = outputIndices[axis] % shape[axis];
            output.values[flatIndex] = values[GetFlatIndex(sourceIndices)];
        }

        return output;
    }

    public Tensor Transpose(params int[] dimensions)
    {
        if (dimensions.Length != Rank)
            throw new ArgumentException("A dimension must be supplied for every tensor axis.", nameof(dimensions));

        var seen = new bool[Rank];
        foreach (var dimension in dimensions)
        {
            if ((uint)dimension >= (uint)Rank || seen[dimension])
                throw new ArgumentException("Dimensions must be a permutation of the tensor axes.", nameof(dimensions));

            seen[dimension] = true;
        }

        var transposedShape = dimensions.Select(dimension => shape[dimension]).ToArray();
        var result = new Tensor(transposedShape);
        var resultIndices = new int[Rank];
        var sourceIndices = new int[Rank];

        for (var flatIndex = 0; flatIndex < result.Length; flatIndex++)
        {
            GetIndices(flatIndex, result.strides, resultIndices);
            for (var axis = 0; axis < Rank; axis++)
                sourceIndices[dimensions[axis]] = resultIndices[axis];

            result.values[flatIndex] = values[GetFlatIndex(sourceIndices)];
        }

        return result;
    }

    public Tensor Add(Tensor other) => Combine(other, static (left, right) => left + right);

    public Tensor Add(float scalar) => Map(value => value + scalar);

    public Tensor Subtract(Tensor other) => Combine(other, static (left, right) => left - right);

    public Tensor Multiply(Tensor other) => Combine(other, static (left, right) => left * right);

    public Tensor Divide(Tensor other) => Combine(other, static (left, right) => left / right);

    public Tensor Multiply(float scalar) => Map(value => value * scalar);

    public Tensor Pow(float exponent) => Map(value => MathF.Pow(value, exponent));

    public Tensor Tanh() => Map(MathF.Tanh);

    public Tensor LessThan(Tensor other) => Combine(other, static (left, right) => left < right ? 1f : 0f);

    public Tensor Sum(int dimension, bool keepdim = false)
    {
        dimension = NormalizeDimension(dimension);
        var outputShape = keepdim
            ? shape.Select((size, axis) => axis == dimension ? 1 : size).ToArray()
            : shape.Where((_, axis) => axis != dimension).ToArray();
        if (outputShape.Length == 0)
            outputShape = [1];

        var output = new Tensor(outputShape);
        var inputIndices = new int[Rank];
        var outputIndices = new int[output.Rank];
        for (var flatIndex = 0; flatIndex < Length; flatIndex++)
        {
            GetIndices(flatIndex, strides, inputIndices);
            if (keepdim)
            {
                Array.Copy(inputIndices, outputIndices, Rank);
                outputIndices[dimension] = 0;
            }
            else
            {
                var outputAxis = 0;
                for (var inputAxis = 0; inputAxis < Rank; inputAxis++)
                    if (inputAxis != dimension)
                        outputIndices[outputAxis++] = inputIndices[inputAxis];
            }

            output.values[output.GetFlatIndex(outputIndices)] += values[flatIndex];
        }

        return output;
    }

    public void copy_(Tensor source)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (!shape.SequenceEqual(source.shape))
            throw new ArgumentException("Tensor shapes must match.", nameof(source));

        Array.Copy(source.values, values, Length);
    }

    public float max_abs_difference(Tensor other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (!shape.SequenceEqual(other.shape))
            throw new ArgumentException("Tensor shapes must match.", nameof(other));

        var maximum = 0f;
        for (var index = 0; index < Length; index++)
            maximum = MathF.Max(maximum, MathF.Abs(values[index] - other.values[index]));
        return maximum;
    }

    public Tensor Divide(float scalar)
    {
        if (scalar == 0f)
            throw new DivideByZeroException();

        return Map(value => value / scalar);
    }

    public Tensor MatMul(Tensor other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (Rank != 2 || other.Rank != 2)
            throw new InvalidOperationException("Matrix multiplication requires two rank-2 tensors.");
        if (shape[1] != other.shape[0])
            throw new ArgumentException("The left tensor column count must match the right tensor row count.", nameof(other));

        var result = new Tensor(shape[0], other.shape[1]);
        for (var row = 0; row < shape[0]; row++)
        {
            for (var column = 0; column < other.shape[1]; column++)
            {
                var sum = 0f;
                for (var index = 0; index < shape[1]; index++)
                    sum += this[row, index] * other[index, column];

                result[row, column] = sum;
            }
        }

        return result;
    }

    public static Tensor operator +(Tensor left, Tensor right) => left.Add(right);

    public static Tensor operator +(Tensor tensor, float scalar) => tensor.Add(scalar);

    public static Tensor operator +(float scalar, Tensor tensor) => tensor.Add(scalar);

    public static Tensor operator -(Tensor left, Tensor right) => left.Subtract(right);

    public static Tensor operator *(Tensor left, Tensor right) => left.Multiply(right);

    public static Tensor operator /(Tensor left, Tensor right) => left.Divide(right);

    public static Tensor operator *(Tensor tensor, float scalar) => tensor.Multiply(scalar);

    public static Tensor operator *(float scalar, Tensor tensor) => tensor.Multiply(scalar);

    public static Tensor operator /(Tensor tensor, float scalar) => tensor.Divide(scalar);

    private Tensor Combine(Tensor other, Func<float, float, float> operation)
    {
        ArgumentNullException.ThrowIfNull(other);
        var outputRank = Math.Max(Rank, other.Rank);
        var outputShape = new int[outputRank];
        for (var axis = 0; axis < outputRank; axis++)
        {
            var leftDimension = axis < outputRank - Rank ? 1 : shape[axis - (outputRank - Rank)];
            var rightDimension = axis < outputRank - other.Rank ? 1 : other.shape[axis - (outputRank - other.Rank)];
            if (leftDimension != rightDimension && leftDimension != 1 && rightDimension != 1)
                throw new ArgumentException("Tensor shapes are not broadcast-compatible.", nameof(other));
            outputShape[axis] = Math.Max(leftDimension, rightDimension);
        }

        var result = new Tensor(outputShape);
        var outputIndices = new int[outputRank];
        var leftIndices = new int[Rank];
        var rightIndices = new int[other.Rank];
        for (var flatIndex = 0; flatIndex < result.Length; flatIndex++)
        {
            GetIndices(flatIndex, result.strides, outputIndices);
            PopulateBroadcastIndices(outputIndices, shape, leftIndices);
            PopulateBroadcastIndices(outputIndices, other.shape, rightIndices);
            result.values[flatIndex] = operation(values[GetFlatIndex(leftIndices)], other.values[other.GetFlatIndex(rightIndices)]);
        }

        return result;
    }

    private Tensor Map(Func<float, float> operation)
    {
        var result = new float[Length];
        for (var index = 0; index < Length; index++)
            result[index] = operation(values[index]);

        return new Tensor(result, shape, takeOwnership: true);
    }

    private int GetFlatIndex(ReadOnlySpan<int> indices)
    {
        if (indices.Length != Rank)
            throw new ArgumentException("The number of indices must match the tensor rank.", nameof(indices));

        var flatIndex = 0;
        for (var axis = 0; axis < Rank; axis++)
        {
            if ((uint)indices[axis] >= (uint)shape[axis])
                throw new ArgumentOutOfRangeException(nameof(indices), "An index is outside the tensor shape.");

            flatIndex += indices[axis] * strides[axis];
        }

        return flatIndex;
    }

    private int NormalizeDimension(int dimension)
    {
        if (dimension < 0)
            dimension += Rank;
        if ((uint)dimension >= (uint)Rank)
            throw new ArgumentOutOfRangeException(nameof(dimension));
        return dimension;
    }

    private static void PopulateBroadcastIndices(ReadOnlySpan<int> outputIndices, IReadOnlyList<int> sourceShape, Span<int> sourceIndices)
    {
        var offset = outputIndices.Length - sourceShape.Count;
        for (var axis = 0; axis < sourceShape.Count; axis++)
            sourceIndices[axis] = sourceShape[axis] == 1 ? 0 : outputIndices[axis + offset];
    }

    private static int[] CreateStrides(IReadOnlyList<int> shape)
    {
        var strides = new int[shape.Count];
        var stride = 1;
        for (var axis = shape.Count - 1; axis >= 0; axis--)
        {
            strides[axis] = stride;
            stride = checked(stride * shape[axis]);
        }

        return strides;
    }

    private static int GetElementCount(IReadOnlyList<int> shape)
    {
        ValidateShape(shape);

        var count = 1;
        foreach (var dimension in shape)
            count = checked(count * dimension);

        return count;
    }

    private static void GetIndices(int flatIndex, IReadOnlyList<int> strides, Span<int> indices)
    {
        for (var axis = 0; axis < strides.Count; axis++)
        {
            indices[axis] = flatIndex / strides[axis];
            flatIndex %= strides[axis];
        }
    }

    private static void ValidateShape(IReadOnlyList<int>? shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (shape.Count == 0)
            throw new ArgumentException("A tensor must have at least one dimension.", nameof(shape));
        if (shape.Any(dimension => dimension <= 0))
            throw new ArgumentOutOfRangeException(nameof(shape), "Tensor dimensions must be positive.");
    }
}

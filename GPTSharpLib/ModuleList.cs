namespace GPTSharpLib;

public sealed class ModuleList<T> : IReadOnlyList<T>
    where T : class
{
    private readonly T[] modules;

    public ModuleList(IEnumerable<T> modules)
    {
        ArgumentNullException.ThrowIfNull(modules);

        this.modules = modules.ToArray();
        if (this.modules.Any(module => module is null))
            throw new ArgumentException("A module list cannot contain null entries.", nameof(modules));
    }

    public int Count => modules.Length;

    public T this[int index] => modules[index];

    public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)modules).GetEnumerator();

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
}

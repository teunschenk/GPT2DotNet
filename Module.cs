public sealed record NamedParameter(string name, Tensor parameter);

public interface IParameterModule
{
    IEnumerable<NamedParameter> named_parameters();
}

public abstract class Module<TInput, TOutput> : IParameterModule
{
    private readonly Dictionary<string, Tensor> registeredParameters = new(StringComparer.Ordinal);

    protected Module(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("A module name is required.", nameof(name));

        Name = name;
    }

    public string Name { get; }

    public bool training { get; private set; } = true;

    public abstract TOutput forward(TInput input);

    public void train() => training = true;

    public void eval() => training = false;

    public IEnumerable<Tensor> parameters() => registeredParameters.Values;

    public IEnumerable<NamedParameter> named_parameters() =>
        registeredParameters.Select(parameter => new NamedParameter(parameter.Key, parameter.Value));

    protected void RegisterParameter(string name, Tensor parameter)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("A parameter name is required.", nameof(name));
        ArgumentNullException.ThrowIfNull(parameter);

        if (!registeredParameters.TryAdd(name, parameter))
            throw new ArgumentException($"A parameter named '{name}' is already registered.", nameof(name));
    }

    protected void RegisterComponents()
    {
        foreach (var field in GetType().GetFields(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic))
        {
            if (field.GetValue(this) is IParameterModule component)
                RegisterComponent(field.Name, component);
            else if (field.GetValue(this) is System.Collections.IEnumerable components)
                foreach (var item in components)
                    if (item is IParameterModule indexedComponent)
                        RegisterComponent($"{field.Name}.{GetComponentIndex(components, item)}", indexedComponent);
        }
    }

    private void RegisterComponent(string name, IParameterModule component)
    {
        foreach (var parameter in component.named_parameters())
            registeredParameters.TryAdd($"{name}.{parameter.name}", parameter.parameter);
    }

    private static int GetComponentIndex(System.Collections.IEnumerable components, object item)
    {
        var index = 0;
        foreach (var candidate in components)
        {
            if (ReferenceEquals(candidate, item))
                return index;
            index++;
        }
        return index;
    }
}

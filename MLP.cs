public class MLP : Module<Tensor, Tensor>
{
    private readonly Linear c_fc;
    private readonly Linear c_proj;

    public MLP(GPTConfig config) : base(nameof(MLP))
    {
        c_fc   = new Linear(config.n_embd, 4 * config.n_embd);
        c_proj = new Linear(4 * config.n_embd, config.n_embd);

        RegisterComponents();
    }

    // MLP.forward
    public override Tensor forward(Tensor x)
    {
        x = c_fc.forward(x);
        var cubic = x.Pow(3f);
        var inner = (x + 0.044715f * cubic) * 0.79788456f;
        x = 0.5f * x * (1f + inner.Tanh());
        x = c_proj.forward(x);
        return x;
    }
}

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class MLP : Module<Tensor, Tensor>
{
    private readonly Linear c_fc;
    private readonly Linear c_proj;

    public MLP(GPTConfig config) : base(nameof(MLP))
    {
        c_fc   = Linear(config.n_embd, 4 * config.n_embd);
        c_proj = Linear(4 * config.n_embd, config.n_embd);

        RegisterComponents();
    }

    // MLP.forward
    public override Tensor forward(Tensor x)
    {
        using var scope = torch.NewDisposeScope();
        x = c_fc.forward(x);
        x = 0.5 * x * (1 + torch.tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * torch.pow(x, 3))));
        x = c_proj.forward(x);
        return x.MoveToOuterDisposeScope();
    }
}

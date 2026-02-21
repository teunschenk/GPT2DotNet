using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class Block : Module<Tensor, Tensor>
{
    private readonly LayerNorm ln_1;
    private readonly CausalSelfAttention attn;
    private readonly LayerNorm ln_2;
    private readonly MLP mlp;

    public Block(GPTConfig config) : base(nameof(Block))
    {
        ln_1 = LayerNorm(config.n_embd);
        attn = new CausalSelfAttention(config);
        ln_2 = LayerNorm(config.n_embd);
        mlp  = new MLP(config);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
       using var scope = torch.NewDisposeScope();
        x = x + attn.forward(ln_1.forward(x));
        x = x + mlp.forward(ln_2.forward(x));
       return x.MoveToOuterDisposeScope();
    }
}

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class CausalSelfAttention : Module<Tensor, Tensor>
{
    private readonly Linear c_attn;
    private readonly Linear c_proj;
    private readonly int n_head;
    private readonly int n_embd;

    public CausalSelfAttention(GPTConfig config) : base(nameof(CausalSelfAttention))
    {
        if (config.n_embd % config.n_head != 0)
            throw new ArgumentException("n_embd must be divisible by n_head");

        // key, query, value projections for all heads, but in a batch
        c_attn = Linear(config.n_embd, 3 * config.n_embd);
        // output projection
        c_proj = Linear(config.n_embd, config.n_embd);

        n_head = config.n_head;
        n_embd = config.n_embd;

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
       using var scope = torch.NewDisposeScope();
        long B = x.size(0); // batch size
        long T = x.size(1); // sequence length
        long C = x.size(2); // embedding dimensionality (n_embd)

        // calculate query, key, values for all heads in batch and move head forward to be the batch dim
        // nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        // e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        var qkv = c_attn.forward(x);
        var splits = qkv.split(n_embd, dim: 2);
        var q = splits[0];
        var k = splits[1];
        var v = splits[2];

        k = k.view(B, T, n_head, C / n_head).transpose(1, 2); // (B, nh, T, hs)
        q = q.view(B, T, n_head, C / n_head).transpose(1, 2); // (B, nh, T, hs)
        v = v.view(B, T, n_head, C / n_head).transpose(1, 2); // (B, nh, T, hs)

        // causal self-attention: apply causal mask then scaled dot-product attention
        
        var y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_casual: true);

        // re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C);

        // output projection
        y = c_proj.forward(y);

       return y.MoveToOuterDisposeScope();
    }
}


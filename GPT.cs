using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions;
using static TorchSharp.torch.nn;

public class GPT : Module<Tensor, Tensor>
{
    private readonly GPTConfig config;

    // transformer components
    private readonly Embedding wte;           // Weight Token Embeddings
    private readonly Embedding wpe;           // Weight Position Embeddings
    private readonly ModuleList<Block> h;     // Hidden Layers
    private readonly LayerNorm ln_f;          // Final Layer Norm

    private readonly Linear lm_head;

    public GPT(GPTConfig config) : base(nameof(GPT))
    {
        this.config = config;

        wte     = Embedding(config.vocab_size, config.n_embd);
        wpe     = Embedding(config.block_size, config.n_embd);
        h       = new ModuleList<Block>(Enumerable.Range(0, config.n_layer).Select(_ => new Block(config)).ToArray());
        ln_f    = LayerNorm(config.n_embd);
        lm_head = Linear(config.n_embd, config.vocab_size, hasBias: false);

        // weight sharing scheme
        wte.weight = lm_head.weight;

        RegisterComponents();
    }

    public override Tensor forward(Tensor idx)
    {
       using var scope = torch.NewDisposeScope();
        long T = idx.size(1);
        if (T > config.block_size)
            throw new ArgumentException($"Cannot forward, model block size is exhausted. Got T={T}, block_size={config.block_size}");

        // shape (T)
        var pos = torch.arange(0, T, dtype: ScalarType.Int64, device: idx.device);

        var pos_emb = wpe.forward(pos);  // shape (T, n_embd)
        var tok_emb = wte.forward(idx);  // shape (B, T, n_embd)
        var x = tok_emb + pos_emb;       // shape (B, T, n_embd)

        foreach (var block in h)
            x = block.forward(x);

        x = ln_f.forward(x);            // shape (B, T, n_embd)
        var logits = lm_head.forward(x); // shape (B, T, vocab_size)

       return logits.MoveToOuterDisposeScope();
    }

    public static GPT from_pretrained(GPT2ModelType modelType)
    {
        var configArgs = new Dictionary<GPT2ModelType, (string huggingFaceId, GPTConfig config)>
        {
            [GPT2ModelType.GPT2]       = ("gpt2",        new(n_layer: 12, n_head: 12, n_embd: 768)),
            [GPT2ModelType.GPT2Medium] = ("gpt2-medium", new(n_layer: 24, n_head: 16, n_embd: 1024)),
            [GPT2ModelType.GPT2Large]  = ("gpt2-large",  new(n_layer: 36, n_head: 20, n_embd: 1280)),
            [GPT2ModelType.GPT2XL]     = ("gpt2-xl",     new(n_layer: 48, n_head: 25, n_embd: 1600)),
        };

        var (model_type, config) = configArgs[modelType];

        Console.WriteLine($"loading weights from pretrained gpt: {model_type}");

        // Download model weights from Hugging Face Hub (cached locally)
        var cachePath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "gpt2dotnet", model_type);
        Directory.CreateDirectory(cachePath);

        var modelPath = Path.Combine(cachePath, "model.safetensors");
        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"Downloading {model_type} weights from Hugging Face...");
            using var httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
            var url = $"https://huggingface.co/{model_type}/resolve/main/model.safetensors";
            using var response = httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();
            using var contentStream = response.Content.ReadAsStreamAsync().GetAwaiter().GetResult();
            using var fileStream = File.Create(modelPath);
            contentStream.CopyTo(fileStream);
            Console.WriteLine("Download complete.");
        }

        // Load HF weights from safetensors file
        var sd_hf = SafetensorsReader.ReadFile(modelPath);

        // Create a from-scratch initialized model
        var model = new GPT(config);
        var sd = model.named_parameters().ToDictionary(p => p.name, p => p.parameter);
        var sd_keys = sd.Keys.Where(k => !k.EndsWith(".attn.bias")).ToList();
       
        //// named_parameters() may deduplicate the tied lm_head.weight; account for it
        if (!sd.ContainsKey("lm_head.weight"))
        {
            sd["lm_head.weight"] = sd["wte.weight"];
        }

        // copy while ensuring all of the parameters are aligned and match in names and shapes
        var sd_keys_hf = sd_hf.Keys
            .Where(k => !k.EndsWith(".attn.masked_bias")) // ignore these, just a buffer
            .Where(k => !k.EndsWith(".attn.bias"))        // same, just the mask (buffer)
            .ToList();
        var transposed = new[] { "attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight" };
        // basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        // this means that we have to transpose these weights when we import them
        if (sd_keys_hf.Count != sd_keys.Count)
            throw new InvalidOperationException($"mismatched keys: {sd_keys_hf.Count} != {sd_keys.Count}");

        using (torch.no_grad())
        {
            foreach (var k in sd_keys_hf)
            {
                // Map HF key to model key: strip "transformer." prefix, keep "lm_head.weight" as-is
                var modelKey = k.StartsWith("transformer.") ? k["transformer.".Length..] : k;

                if (transposed.Any(t => k.EndsWith(t)))
                {
                    // special treatment for the Conv1D weights we need to transpose
                    if (!sd_hf[k].shape.Reverse().SequenceEqual(sd[modelKey].shape))
                        throw new InvalidOperationException($"transposed shape mismatch for {k}");
                    sd[modelKey].copy_(sd_hf[k].t());
                }
                else
                {
                    // vanilla copy over the other parameters
                    if (!sd_hf[k].shape.SequenceEqual(sd[modelKey].shape))
                        throw new InvalidOperationException($"shape mismatch for {k}");
                    sd[modelKey].copy_(sd_hf[k]);
                }

                sd_hf[k].Dispose();
            }
        }

        // Verify that the model parameters were loaded correctly from the checkpoint.
        // Re-read the safetensors file for a spot-check comparison.
        var sd_verify = SafetensorsReader.ReadFile(modelPath);
        using (torch.no_grad())
        {
            foreach (var k in sd_keys_hf)
            {
                var modelKey = k.StartsWith("transformer.") ? k["transformer.".Length..] : k;
                var modelParam = sd[modelKey];

                Tensor expected;
                if (transposed.Any(t => k.EndsWith(t)))
                    expected = sd_verify[k].t();
                else
                    expected = sd_verify[k];

                if (!modelParam.shape.SequenceEqual(expected.shape))
                    throw new InvalidOperationException(
                        $"Verification failed: shape mismatch for '{modelKey}'. " +
                        $"Model: [{string.Join(", ", modelParam.shape)}], " +
                        $"Expected: [{string.Join(", ", expected.shape)}]");

                // allclose checks element-wise equality within tolerance
                if (!torch.allclose(modelParam.to(expected.dtype), expected, rtol: 1e-3, atol: 1e-5))
                    throw new InvalidOperationException(
                        $"Verification failed: value mismatch for '{modelKey}'. " +
                        $"Max diff: {(modelParam.to(expected.dtype) - expected).abs().max().item<float>()}");

                expected.Dispose();
                sd_verify[k].Dispose();
            }
        }
        Console.WriteLine("Verification passed: all loaded weights match the checkpoint.");

        return model;
    }
}

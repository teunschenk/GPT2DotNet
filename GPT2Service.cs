using System.Diagnostics;
using Tiktoken;
using TorchSharp;

public class GPT2Service
{
    private readonly GPT model;
    private readonly Encoder encoder;

    public GPT2Service(GPT model, Encoder encoder)
    {
        this.model = model;
        this.encoder = encoder;
    }

    public static GPT LoadModel(GPT2ModelType gpt2Model)
    {
        Console.WriteLine("Initializing GPT model from pretrained gpt2 weights...");
        var model = GPT.from_pretrained(gpt2Model);
        Console.WriteLine("Loaded pretrained gpt2 model successfully!");
        Console.WriteLine($"Number of parameters: {model.parameters().ToList().Count()}");

        model.eval();
        model.to(torch.CUDA);
        return model;
    }

    public string GenerateText(int numSequences, int sequenceLength, string input)
    {
        var tokens = encoder.Encode(input);
        Console.WriteLine(input);

        var tokensTensor = torch.tensor(tokens.Select(t => (long)t).ToArray(), dtype: torch.int64, device: torch.CUDA);
        var x = tokensTensor.unsqueeze(0).repeat(numSequences, 1); // (5, 8)

        torch.manual_seed(1337);
        torch.cuda.manual_seed(1337);

        var sw = Stopwatch.StartNew();

        using (torch.no_grad())
        {
            for (int i = 0; i < sequenceLength; i++)
            {
                using (var scope = torch.NewDisposeScope())
                {
                    var logits = model.forward(x);                                          // (5, T, 50257)
                    logits = logits[.., -1, ..];                                            // (5, 50257)
                    var probs = torch.nn.functional.softmax(logits, dim: -1);               // (5, 50257)
                    var (topk_probs, topk_indices) = torch.topk(probs, k: 50, dim: -1);    // (5, 50)
                    var ix = torch.multinomial(topk_probs, num_samples: 1);                 // (5, 1)
                    var xcol = torch.gather(topk_indices, dim: -1, index: ix);              // (5, 1)
                    x = torch.cat([x, xcol], dim: 1);                                      // (5, 9), (5, 10), ...
                    x.MoveToOuterDisposeScope();
                    //print the newly generated token
                    var newToken = xcol[0][0].item<long>();
                    Console.Write($"{encoder.Decode(new List<int> { (int)newToken })}");
                }
            }
        }

        sw.Stop();
        var totalTokens = numSequences * sequenceLength;
        var tokensPerSecond = totalTokens / sw.Elapsed.TotalSeconds;
        Console.WriteLine();
        Console.WriteLine($"Generated {totalTokens} tokens in {sw.Elapsed.TotalSeconds:F2}s ({tokensPerSecond:F2} tokens/sec)");

        var row = x[0];
        var generated = Enumerable.Range(0, (int)row.size(0)).Select(j => (int)row[j].item<long>()).ToList();
        return encoder.Decode(generated);
    }
}

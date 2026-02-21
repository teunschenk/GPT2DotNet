using System.Diagnostics;
using Tiktoken;
using TorchSharp;
using System.Linq;

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

    public string GenerateText(int sequenceLength, string input)
    {
        var numSequences = 1;
        var tokens = encoder.Encode(input);
        Console.WriteLine(input);

        var tokensTensor = torch.tensor(tokens.Select(t => (long)t).ToArray(), dtype: torch.int64, device: torch.CUDA);
        var x = tokensTensor.unsqueeze(0).repeat(numSequences, 1); // (5, 8)

        torch.manual_seed(1337);
        torch.cuda.manual_seed(1337);

        string[] stopTokens = { "\n", "\nQ:", "\nUser", "<|endoftext|>" };

        var sw = Stopwatch.StartNew();

        using (torch.no_grad())
        {
            for (int i = 0; i < sequenceLength; i++)
            {
                using (var scope = torch.NewDisposeScope())
                {
                    var logits = model.forward(x);                                          // (B, T, 50257)
                    logits = logits[.., -1, ..];                                            // (B, 50257)

                    // Repetition penalty: penalize tokens already in the sequence
                    var generatedTokens = new HashSet<long>();
                    for (int t = 0; t < (int)x.size(1); t++)
                        generatedTokens.Add(x[0][t].item<long>());
                    foreach (var token in generatedTokens)
                    {
                        var score = logits[0][(int)token].item<float>();
                        logits[0][(int)token] = score > 0 ? score / 1.2f : score * 1.2f;
                    }

                    // No-repeat n-gram (size 3): prevent repeating trigrams
                    int ngramSize = 3;
                    int seqLen = (int)x.size(1);
                    if (seqLen >= ngramSize - 1)
                    {
                        var bannedTokens = new HashSet<long>();
                        for (int j = 0; j <= seqLen - ngramSize; j++)
                        {
                            bool matchesSuffix = true;
                            for (int k = 0; k < ngramSize - 1; k++)
                            {
                                if (x[0][j + k].item<long>() != x[0][seqLen - (ngramSize - 1) + k].item<long>())
                                {
                                    matchesSuffix = false;
                                    break;
                                }
                            }
                            if (matchesSuffix)
                                bannedTokens.Add(x[0][j + ngramSize - 1].item<long>());
                        }
                        foreach (var token in bannedTokens)
                            logits[0][(int)token] = float.NegativeInfinity;
                    }

                    // Temperature scaling
                    logits = logits / 0.5f;

                    // Top-k filtering: keep only top 40 tokens to remove long-tail noise
                    int topK = 40;
                    var (topkValues, _) = torch.topk(logits, topK, dim: -1);
                    var minTopK = topkValues[.., -1].unsqueeze(-1);
                    logits = torch.where(logits < minTopK, torch.tensor(float.NegativeInfinity, device: logits.device), logits);

                    var probs = torch.nn.functional.softmax(logits, dim: -1);               // (B, 50257)

                    // Top-p (nucleus) sampling with p=0.7
                    var (sortedProbs, sortedIndices) = torch.sort(probs, dim: -1, descending: true);
                    var cumulativeProbs = torch.cumsum(sortedProbs, dim: -1);
                    var sortedMask = cumulativeProbs - sortedProbs > 0.7f;
                    sortedProbs[sortedMask] = 0.0f;
                    sortedProbs = sortedProbs / sortedProbs.sum(dim: -1, keepdim: true);
                    var ix = torch.multinomial(sortedProbs, num_samples: 1);                 // (B, 1)
                    var xcol = torch.gather(sortedIndices, dim: -1, index: ix);              // (B, 1)
                    x = torch.cat([x, xcol], dim: 1);                                      // (5, 9), (5, 10), ...
                    x.MoveToOuterDisposeScope();                    

                    // Check if any of the generated tokens is a stop token based on the last 10 tokens in the sequence
                    var lastTokensObj = x[0][^10..].tolist();
                    var lastTokensList = ((System.Collections.ArrayList)lastTokensObj)
                        .Cast<TorchSharp.Scalar>()
                        .Select(s => s.ToInt64())
                        .ToList();
                    string lastTokensStr = string.Join("", lastTokensList.Select(t => encoder.Decode(new List<int> { (int)t })));
                    if (stopTokens.Any(stop => lastTokensStr.Contains(stop)))
                    {                        
                        Console.WriteLine("\nStop token detected. Ending generation.");
                        break;
                    }

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

using System.Diagnostics;
using Tiktoken;
using System.Linq;

namespace GPTSharpLib;

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
        //model.to(TensorOperations.CUDA);
        return model;
    }

    public string GenerateText(int sequenceLength, string input)
    {
        var numSequences = 1;
        var tokens = encoder.Encode(input);
        Console.WriteLine(input);

        var x = new Tensor(tokens.Select(token => (float)token).ToArray(), 1, tokens.Count).repeat(numSequences, 1);

        TensorOperations.manual_seed(1337);

        string[] stopTokens = { "\n", "\nQ:", "\nUser", "<|endoftext|>" };

        var sw = Stopwatch.StartNew();

        for (var i = 0; i < sequenceLength; i++)
        {
            var logits = model.forward(x);
            var vocabularySize = logits.shape[2];
            var sequenceWidth = x.shape[1];
            logits = logits.Slice(1, sequenceWidth - 1, 1).Reshape(numSequences, vocabularySize);

            for (var batch = 0; batch < numSequences; batch++)
            {
                var generatedTokens = new HashSet<int>();
                for (var tokenIndex = 0; tokenIndex < sequenceWidth; tokenIndex++)
                    generatedTokens.Add((int)x[batch, tokenIndex]);
                foreach (var token in generatedTokens)
                {
                    var score = logits[batch, token];
                    logits[batch, token] = score > 0 ? score / 1.2f : score * 1.2f;
                }

                const int ngramSize = 3;
                if (sequenceWidth >= ngramSize)
                for (var start = 0; start <= sequenceWidth - ngramSize; start++)
                {
                    var matchesSuffix = true;
                    for (var offset = 0; offset < ngramSize - 1; offset++)
                    {
                        if (x[batch, start + offset] != x[batch, sequenceWidth - (ngramSize - 1) + offset])
                        {
                            matchesSuffix = false;
                            break;
                        }
                    }
                    if (matchesSuffix)
                        logits[batch, (int)x[batch, start + ngramSize - 1]] = float.NegativeInfinity;
                }
            }

            logits = logits / 0.5f;
            var (topKValues, _) = TensorOperations.topk(logits, 40, dim: -1);
            for (var batch = 0; batch < numSequences; batch++)
            {
                var threshold = topKValues[batch, 39];
                for (var token = 0; token < vocabularySize; token++)
                    if (logits[batch, token] < threshold)
                        logits[batch, token] = float.NegativeInfinity;
            }

            var probabilities = TensorOperations.nn.functional.softmax(logits, dim: -1);
            var nextTokens = TensorOperations.multinomial(probabilities, num_samples: 1);
            x = TensorOperations.cat([x, nextTokens], dim: 1);

            var recentStart = Math.Max(0, x.shape[1] - 10);
            var recentText = string.Concat(Enumerable.Range(recentStart, x.shape[1] - recentStart)
                .Select(index => encoder.Decode(new List<int> { (int)x[0, index] })));
            if (stopTokens.Any(stop => recentText.Contains(stop)))
            {
                Console.WriteLine("\nStop token detected. Ending generation.");
                break;
            }

            Console.Write(encoder.Decode(new List<int> { (int)nextTokens[0, 0] }));
        }

        sw.Stop();
        var totalTokens = numSequences * sequenceLength;
        var tokensPerSecond = totalTokens / sw.Elapsed.TotalSeconds;
        Console.WriteLine();
        Console.WriteLine($"Generated {totalTokens} tokens in {sw.Elapsed.TotalSeconds:F2}s ({tokensPerSecond:F2} tokens/sec)");

        var generated = Enumerable.Range(0, x.shape[1]).Select(index => (int)x[0, index]).ToList();
        return encoder.Decode(generated);
    }
}

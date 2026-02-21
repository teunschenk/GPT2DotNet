public enum GPT2ModelType
{
    GPT2,
    GPT2Medium,
    GPT2Large,
    GPT2XL,
}

public record GPTConfig(
    int block_size = 1024,  // max sequence length
    int vocab_size = 50257, // number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    int n_layer    = 12,    // number of layers
    int n_head     = 12,    // number of heads
    int n_embd     = 768    // embedding dimension
);


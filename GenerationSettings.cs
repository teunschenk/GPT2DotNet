public class GenerationSettings
{
    public float Temperature { get; set; } = 0.5f;
    public int TopK { get; set; } = 40;
    public float TopP { get; set; } = 0.7f;
    public float RepetitionPenalty { get; set; } = 1.2f;
    public int NgramSize { get; set; } = 3;
    public int MaxTokens { get; set; } = 60;
}

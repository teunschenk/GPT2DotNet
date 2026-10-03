using System.Text;
using GPTSharpLib;
using Tiktoken;
using Tiktoken.Encodings;

var model = GPT2Service.LoadModel(GPT2ModelType.GPT2);
var service = new GPT2Service(model, new global::Tiktoken.Encoder(new R50KBase()));
var messages = new List<ChatMessage>();

Console.WriteLine("Chat ready. Type 'exit' or 'quit' to end the session.");

while (true)
{
    Console.Write("You: ");
    var input = Console.ReadLine();
    if (input is null)
        break;

    input = input.Trim();
    if (string.IsNullOrEmpty(input))
        continue;

    if (string.Equals(input, "exit", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(input, "quit", StringComparison.OrdinalIgnoreCase))
        break;

    Console.Write("Thinking...");

    messages.Add(new ChatMessage("User", input));
    var prompt = BuildPrompt(messages);

    Console.Write("Assistant: ");
    var output = service.GenerateText(60, prompt);
    messages.Add(new ChatMessage("Assistant", ExtractAssistantReply(output, prompt)));

    Console.WriteLine(messages.Last().Text);
}

static string BuildPrompt(IEnumerable<ChatMessage> messages)
{
    var prompt = new StringBuilder();
    prompt.AppendLine("You are a knowledgeable assistant.");
    prompt.AppendLine("Answer the question clearly and in one short sentence.");
    prompt.AppendLine();

    foreach (var message in messages)
        prompt.AppendLine(message.Role == "User" ? $"Q: {message.Text}" : $"A: {message.Text}");

    prompt.Append("A:");
    return prompt.ToString();
}

static string ExtractAssistantReply(string fullOutput, string prompt)
{
    var reply = fullOutput.Length > prompt.Length ? fullOutput[prompt.Length..] : fullOutput;
    var stopMarkers = new[] { "\nQ:", "\nUser" };
    var earliest = reply.Length;

    foreach (var marker in stopMarkers)
    {
        var index = reply.IndexOf(marker, StringComparison.Ordinal);
        if (index >= 0 && index < earliest)
            earliest = index;
    }

    return reply[..earliest].Trim();
}

sealed record ChatMessage(string Role, string Text);

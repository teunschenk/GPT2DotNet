using System.Text;
using Microsoft.AspNetCore.Components;

namespace GPT2DotNet.Components.Pages;

public partial class Chat : ComponentBase
{
    [Inject]
    private GPT2Service Gpt2 { get; set; } = default!;

    protected sealed record ChatMessage(string Role, string Text);

    protected readonly List<ChatMessage> _messages = [];
    protected string _userInput = string.Empty;
    protected bool _generating;

    private const int SequenceLength = 100;

    protected void Clear()
    {
        _messages.Clear();
    }

    protected async Task Send()
    {
        var text = _userInput.Trim();
        if (string.IsNullOrEmpty(text))
            return;

        _messages.Add(new ChatMessage("User", text));
        _userInput = string.Empty;
        _generating = true;
        StateHasChanged();

        var prompt = BuildPrompt();

        var result = await Task.Run(() => Gpt2.GenerateText(SequenceLength, prompt));

        var reply = ExtractAssistantReply(result, prompt);
        _messages.Add(new ChatMessage("Assistant", reply));
        _generating = false;
    }

    private string BuildPrompt()
    {
        var sb = new StringBuilder();
        sb.AppendLine("The following is a conversation between a helpful AI assistant and a user.");
        sb.AppendLine("The assistant gives direct, factual, concise answers and does not ask unnecessary questions.");
        sb.AppendLine();

        foreach (var msg in _messages)
        {
            sb.AppendLine($"{msg.Role}: {msg.Text}");
        }

        sb.Append("Assistant:");
        return sb.ToString();
    }

    private static string ExtractAssistantReply(string fullOutput, string prompt)
    {
        var reply = fullOutput.Length > prompt.Length
            ? fullOutput[prompt.Length..]
            : fullOutput;

        var stopIdx = reply.IndexOf("\nUser", StringComparison.Ordinal);
        if (stopIdx >= 0)
            reply = reply[..stopIdx];

        return reply.Trim();
    }
}

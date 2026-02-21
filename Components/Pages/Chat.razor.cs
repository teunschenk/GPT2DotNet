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

    private const int SequenceLength = 60;

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
        sb.AppendLine("You are a knowledgeable assistant.");
        sb.AppendLine("Answer the question clearly and in one short sentence.");
        sb.AppendLine();

        foreach (var msg in _messages)
        {
            if (msg.Role == "User")
                sb.AppendLine($"Q: {msg.Text}");
            else
                sb.AppendLine($"A: {msg.Text}");
        }

        sb.Append("A:");
        return sb.ToString();
    }

    private static string ExtractAssistantReply(string fullOutput, string prompt)
    {
        var reply = fullOutput.Length > prompt.Length
            ? fullOutput[prompt.Length..]
            : fullOutput;

        // Stop at the earliest occurrence of any stop marker
        string[] stopMarkers = ["\nQ:", "\nUser"];
        int earliest = reply.Length;
        foreach (var marker in stopMarkers)
        {
            var idx = reply.IndexOf(marker, StringComparison.Ordinal);
            if (idx >= 0 && idx < earliest)
                earliest = idx;
        }
        reply = reply[..earliest];

        return reply.Trim();
    }
}

using Tiktoken;
using Tiktoken.Encodings;
using GPT2DotNet.Components;
using GPTSharpLib;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

builder.Services.AddSingleton(_ =>
{
    var model = GPT2Service.LoadModel(GPT2ModelType.GPT2);
    var enc = new Encoder(new R50KBase());
    return new GPT2Service(model, enc);
});

var app = builder.Build();

app.UseStaticFiles();
app.UseAntiforgery();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();


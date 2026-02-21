// Entry point
using Tiktoken;
using Tiktoken.Encodings;

Console.WriteLine("Hello, GPT2!");

var model = GPT2Service.LoadModel(GPT2ModelType.GPT2);
var enc = new Encoder(new R50KBase()); //GPT2 encoding
var service = new GPT2Service(model, enc);

const int NUM_SEQUENCES = 1;
const int SEQUENCE_LENGTH = 100;

var input = "The following is a conversation between a helpful AI assistant and a user." +
    "\nThe assistant gives direct, factual, concise answers and does not ask unnecessary questions." +
    "\n" +
    "\nUser: Name the capital of The Netherlands!" +
    "\nAssistant:";

var result = service.GenerateText(NUM_SEQUENCES, SEQUENCE_LENGTH, input);

//Console.WriteLine("--------------------------\nGenerated text:");
//Console.WriteLine(result);

//// Print the generated text
//for (int i = 0; i < NUM_SEQUENCES; i++)
//{
//    var row = x[i];
//    var generated = Enumerable.Range(0, (int)row.size(0)).Select(j => (int)row[j].item<long>()).ToList();
//    Console.WriteLine($"Result {i}>>> {enc.Decode(generated)}<<<");
//}


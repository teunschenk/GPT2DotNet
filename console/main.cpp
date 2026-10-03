#include "GPTPlusLib.hpp"

#include <iostream>
#include <string>

namespace
{
    void PrintUsage()
    {
        std::cerr << "Usage: console <model.safetensors> <encoder.json> <vocab.bpe> [gpt2|medium|large|xl]\n";
    }

    gptplus::GPT2ModelType ParseModelType(const std::string& value)
    {
        if (value == "gpt2") return gptplus::GPT2ModelType::GPT2;
        if (value == "medium") return gptplus::GPT2ModelType::GPT2Medium;
        if (value == "large") return gptplus::GPT2ModelType::GPT2Large;
        if (value == "xl") return gptplus::GPT2ModelType::GPT2XL;
        throw std::invalid_argument("Unknown GPT-2 model type: " + value);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 4 || argc > 5)
    {
        PrintUsage();
        return 1;
    }

    try
    {
		std::cout << "Loading GPT-2 model from: " << argv[1] << "\n";
        const auto type = argc == 5 ? ParseModelType(argv[4]) : gptplus::GPT2ModelType::GPT2;
        auto tokenizer = gptplus::GPT2Tokenizer{};
        tokenizer.load(argv[2], argv[3]);
        auto service = gptplus::GPT2Service(gptplus::GPT::load(argv[1], type), std::move(tokenizer));

        std::cout << "Chat ready. Type 'exit' or 'quit' to end the session.\n";
        for (std::string input; ; )
        {
            std::cout << "You: ";
            if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;
            if (input.empty()) continue;
            const auto prompt = "You are a knowledgeable assistant.\nAnswer the question clearly and in one short sentence.\n\nQ: " + input + "\nA:";
            const auto output = service.generateText(prompt);
            std::cout << "Assistant: " << output.substr(prompt.size()) << "\n";
        }
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "GPTPlusLib error: " << exception.what() << "\n";
        return 1;
    }
}

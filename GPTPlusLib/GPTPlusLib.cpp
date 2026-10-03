#include "GPTPlusLib.hpp"
#include <cctype>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <set>
#include <sstream>

namespace gptplus
{
    struct GPT::Implementation
    {
        explicit Implementation(GPTConfig value) : configuration(std::move(value)) {}
        GPTConfig configuration;
        std::unordered_map<std::string, Tensor> parameters;
    };

    namespace
    {
        const Tensor& Parameter(const GPT::Implementation& implementation, const std::string& name)
        {
            const auto iterator = implementation.parameters.find(name);
            if (iterator == implementation.parameters.end())
                throw std::logic_error("Model parameter is missing: " + name);
            return iterator->second;
        }

        Tensor LinearForward(const Tensor& input, const Tensor& weight, const Tensor* bias)
        {
            if (weight.rank() != 2 || input.shape().back() != weight.shape()[1])
                throw std::invalid_argument("Linear parameter shape does not match its input.");
            auto outputShape = input.shape();
            outputShape.back() = weight.shape()[0];
            Tensor output(outputShape);
            const auto inputWidth = weight.shape()[1];
            const auto outputWidth = weight.shape()[0];
            const auto groupCount = input.length() / inputWidth;
            const auto& inputValues = input.values();
            const auto& weightValues = weight.values();
            const auto* biasValues = bias == nullptr ? nullptr : &bias->values();
            auto& outputValues = output.values();
#pragma omp parallel for collapse(2) schedule(static)
            for (int group = 0; group < groupCount; ++group)
            for (int outputIndex = 0; outputIndex < outputWidth; ++outputIndex)
            {
                auto value = biasValues == nullptr ? 0.0F : (*biasValues)[outputIndex];
                for (int inputIndex = 0; inputIndex < inputWidth; ++inputIndex)
                    value += inputValues[group * inputWidth + inputIndex] * weightValues[outputIndex * inputWidth + inputIndex];
                outputValues[group * outputWidth + outputIndex] = value;
            }
            return output;
        }

        Tensor LayerNormForward(const Tensor& input, const Tensor& weight, const Tensor& bias)
        {
            const auto width = input.shape().back();
            Tensor output(input.shape());
            for (int group = 0; group < input.length() / width; ++group)
            {
                const auto offset = group * width;
                float mean = 0.0F;
                for (int index = 0; index < width; ++index) mean += input.values()[offset + index];
                mean /= width;
                float variance = 0.0F;
                for (int index = 0; index < width; ++index) { const auto difference = input.values()[offset + index] - mean; variance += difference * difference; }
                const auto inverseStandardDeviation = 1.0F / std::sqrt(variance / width + 1e-5F);
                for (int index = 0; index < width; ++index)
                    output.values()[offset + index] = (input.values()[offset + index] - mean) * inverseStandardDeviation * weight.values()[index] + bias.values()[index];
            }
            return output;
        }

        Tensor Gelu(const Tensor& input)
        {
            Tensor output(input.shape());
            for (int index = 0; index < input.length(); ++index)
            {
                const auto value = input.values()[index];
                output.values()[index] = 0.5F * value * (1.0F + std::tanh(0.79788456F * (value + 0.044715F * value * value * value)));
            }
            return output;
        }

        Tensor Add(const Tensor& left, const Tensor& right) { return left.add(right); }

        Tensor Attention(const Tensor& input, const GPT::Implementation& model, int layer)
        {
            const auto prefix = "h." + std::to_string(layer) + ".attn.";
            const auto& projectionWeight = Parameter(model, prefix + "c_attn.weight");
            const auto& projectionBias = Parameter(model, prefix + "c_attn.bias");
            const auto qkv = LinearForward(input, projectionWeight, &projectionBias);
            const auto batch = input.shape()[0];
            const auto tokens = input.shape()[1];
            const auto channels = input.shape()[2];
            const auto heads = model.configuration.headCount;
            const auto headSize = channels / heads;
            Tensor attentionOutput({ batch, tokens, channels });
            for (int b = 0; b < batch; ++b)
            for (int h = 0; h < heads; ++h)
            for (int target = 0; target < tokens; ++target)
            {
                std::vector<float> scores(target + 1);
                float maximum = -std::numeric_limits<float>::infinity();
                for (int source = 0; source <= target; ++source)
                {
                    float score = 0.0F;
                    for (int channel = 0; channel < headSize; ++channel)
                    {
                        const auto queryIndex = ((b * tokens + target) * 3 * channels) + h * headSize + channel;
                        const auto keyIndex = ((b * tokens + source) * 3 * channels) + channels + h * headSize + channel;
                        score += qkv.values()[queryIndex] * qkv.values()[keyIndex];
                    }
                    scores[source] = score / std::sqrt(static_cast<float>(headSize));
                    maximum = std::max(maximum, scores[source]);
                }
                float sum = 0.0F;
                for (auto& score : scores) { score = std::exp(score - maximum); sum += score; }
                for (int source = 0; source <= target; ++source)
                for (int channel = 0; channel < headSize; ++channel)
                {
                    const auto valueIndex = ((b * tokens + source) * 3 * channels) + 2 * channels + h * headSize + channel;
                    attentionOutput.values()[((b * tokens + target) * channels) + h * headSize + channel] += scores[source] / sum * qkv.values()[valueIndex];
                }
            }

            return LinearForward(attentionOutput, Parameter(model, prefix + "c_proj.weight"), &Parameter(model, prefix + "c_proj.bias"));
        }

        GPTConfig ConfigFor(GPT2ModelType type)
        {
            switch (type)
            {
            case GPT2ModelType::GPT2: return {};
            case GPT2ModelType::GPT2Medium: return { 1024, 50257, 24, 16, 1024 };
            case GPT2ModelType::GPT2Large: return { 1024, 50257, 36, 20, 1280 };
            case GPT2ModelType::GPT2XL: return { 1024, 50257, 48, 25, 1600 };
            }
            throw std::invalid_argument("Unsupported GPT-2 model type.");
        }

        std::vector<int> ParseShape(const std::string& shapeText)
        {
            std::vector<int> shape;
            static const std::regex number("[0-9]+");
            for (std::sregex_iterator match(shapeText.begin(), shapeText.end(), number), end; match != end; ++match)
                shape.push_back(std::stoi(match->str()));
            return shape;
        }

        bool IsTransposedWeight(const std::string& name)
        {
            const auto endsWith = [&name](const char* suffix)
            {
                const std::string value(suffix);
                return name.size() >= value.size() && name.compare(name.size() - value.size(), value.size(), value) == 0;
            };

            return endsWith("attn.c_attn.weight") ||
                endsWith("attn.c_proj.weight") ||
                endsWith("mlp.c_fc.weight") ||
                endsWith("mlp.c_proj.weight");
        }

        Tensor TransposeWeight(const Tensor& input)
        {
            if (input.rank() != 2)
                throw std::invalid_argument("A GPT-2 Conv1D weight must be rank 2.");
            return input.transpose(0, 1);
        }

        void AppendUtf8(std::string& output, int codePoint)
        {
            if (codePoint <= 0x7f)
                output.push_back(static_cast<char>(codePoint));
            else if (codePoint <= 0x7ff)
            {
                output.push_back(static_cast<char>(0xc0 | (codePoint >> 6)));
                output.push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
            }
            else
            {
                output.push_back(static_cast<char>(0xe0 | (codePoint >> 12)));
                output.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3f)));
                output.push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
            }
        }

        int HexValue(char value)
        {
            if (value >= '0' && value <= '9') return value - '0';
            if (value >= 'a' && value <= 'f') return value - 'a' + 10;
            if (value >= 'A' && value <= 'F') return value - 'A' + 10;
            throw std::runtime_error("Invalid JSON Unicode escape in GPT-2 vocabulary.");
        }

        std::string JsonUnescape(const std::string& value)
        {
            std::string result;
            for (std::size_t index = 0; index < value.size(); ++index)
            {
                if (value[index] != '\\')
                {
                    result.push_back(value[index]);
                    continue;
                }
                if (++index == value.size())
                    throw std::runtime_error("Invalid JSON escape in GPT-2 vocabulary.");
                if (value[index] != 'u')
                {
                    result.push_back(value[index]);
                    continue;
                }
                if (index + 4 >= value.size())
                    throw std::runtime_error("Incomplete JSON Unicode escape in GPT-2 vocabulary.");
                auto codePoint = 0;
                for (int digit = 0; digit < 4; ++digit)
                    codePoint = (codePoint << 4) | HexValue(value[++index]);
                AppendUtf8(result, codePoint);
            }
            return result;
        }

        std::vector<std::string> Utf8Pieces(const std::string& value)
        {
            std::vector<std::string> result;
            for (std::size_t index = 0; index < value.size(); )
            {
                const auto first = static_cast<unsigned char>(value[index]);
                const auto length = first < 0x80 ? 1 : first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4;
                if (index + length > value.size())
                    throw std::runtime_error("Invalid UTF-8 in GPT-2 tokenizer data.");
                result.push_back(value.substr(index, length));
                index += length;
            }
            return result;
        }

        const std::array<std::string, 256>& ByteEncoder()
        {
            static const auto encoder = []
            {
                std::array<std::string, 256> result;
                std::vector<int> bytes;
                for (int value = 33; value <= 126; ++value) bytes.push_back(value);
                for (int value = 161; value <= 172; ++value) bytes.push_back(value);
                for (int value = 174; value <= 255; ++value) bytes.push_back(value);
                auto codePoints = bytes;
                for (int value = 0, next = 0; value < 256; ++value)
                    if (std::find(bytes.begin(), bytes.end(), value) == bytes.end())
                    {
                        bytes.push_back(value);
                        codePoints.push_back(256 + next++);
                    }
                for (std::size_t index = 0; index < bytes.size(); ++index)
                    AppendUtf8(result[bytes[index]], codePoints[index]);
                return result;
            }();
            return encoder;
        }

        const std::unordered_map<std::string, unsigned char>& ByteDecoder()
        {
            static const auto decoder = []
            {
                std::unordered_map<std::string, unsigned char> result;
                const auto& encoder = ByteEncoder();
                for (int value = 0; value < 256; ++value)
                    result.emplace(encoder[value], static_cast<unsigned char>(value));
                return result;
            }();
            return decoder;
        }

        std::vector<std::string> Bpe(const std::string& token, const std::unordered_map<std::string, int>& ranks)
        {
            auto parts = Utf8Pieces(token);
            while (parts.size() > 1)
            {
                int bestIndex = -1;
                auto bestRank = std::numeric_limits<int>::max();
                for (int index = 0; index + 1 < static_cast<int>(parts.size()); ++index)
                {
                    const auto iterator = ranks.find(parts[index] + " " + parts[index + 1]);
                    if (iterator != ranks.end() && iterator->second < bestRank) { bestRank = iterator->second; bestIndex = index; }
                }
                if (bestIndex < 0) break;
                parts[bestIndex] += parts[bestIndex + 1];
                parts.erase(parts.begin() + bestIndex + 1);
            }
            return parts;
        }
    }

    GPT::GPT(GPTConfig config) : implementation_(std::make_shared<Implementation>(std::move(config))) {}
    const GPTConfig& GPT::config() const noexcept { return implementation_->configuration; }

    Tensor GPT::forward(const Tensor& tokenIds) const
    {
        if (tokenIds.rank() != 2)
            throw std::invalid_argument("GPT token IDs must have shape [batch, tokens].");
        const auto batch = tokenIds.shape()[0];
        const auto tokens = tokenIds.shape()[1];
        const auto& configuration = implementation_->configuration;
        if (tokens > configuration.blockSize)
            throw std::invalid_argument("The input sequence exceeds the model block size.");
        const auto& tokenEmbedding = Parameter(*implementation_, "wte.weight");
        const auto& positionEmbedding = Parameter(*implementation_, "wpe.weight");
        Tensor hidden({ batch, tokens, configuration.embeddingSize });
        for (int b = 0; b < batch; ++b)
        for (int token = 0; token < tokens; ++token)
        {
            const auto tokenId = static_cast<int>(tokenIds.values()[b * tokens + token]);
            if (tokenId < 0 || tokenId >= configuration.vocabSize)
                throw std::out_of_range("Token ID is outside the vocabulary.");
            for (int channel = 0; channel < configuration.embeddingSize; ++channel)
                hidden.values()[(b * tokens + token) * configuration.embeddingSize + channel] = tokenEmbedding.values()[tokenId * configuration.embeddingSize + channel] + positionEmbedding.values()[token * configuration.embeddingSize + channel];
        }
        for (int layer = 0; layer < configuration.layerCount; ++layer)
        {
            const auto prefix = "h." + std::to_string(layer) + ".";
            const auto normalized = LayerNormForward(hidden, Parameter(*implementation_, prefix + "ln_1.weight"), Parameter(*implementation_, prefix + "ln_1.bias"));
            hidden = Add(hidden, Attention(normalized, *implementation_, layer));
            const auto normalizedMlp = LayerNormForward(hidden, Parameter(*implementation_, prefix + "ln_2.weight"), Parameter(*implementation_, prefix + "ln_2.bias"));
            const auto mlpInput = LinearForward(normalizedMlp, Parameter(*implementation_, prefix + "mlp.c_fc.weight"), &Parameter(*implementation_, prefix + "mlp.c_fc.bias"));
            hidden = Add(hidden, LinearForward(Gelu(mlpInput), Parameter(*implementation_, prefix + "mlp.c_proj.weight"), &Parameter(*implementation_, prefix + "mlp.c_proj.bias")));
        }
        hidden = LayerNormForward(hidden, Parameter(*implementation_, "ln_f.weight"), Parameter(*implementation_, "ln_f.bias"));
        return LinearForward(hidden, tokenEmbedding, nullptr);
    }

    GPT GPT::load(const std::string& safetensorsPath, GPT2ModelType type)
    {
        std::ifstream stream(safetensorsPath, std::ios::binary);
        if (!stream)
            throw std::runtime_error("Unable to open safetensors file: " + safetensorsPath);
        std::uint64_t headerSize = 0;
        stream.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize));
        if (!stream || headerSize == 0 || headerSize > 64ULL * 1024ULL * 1024ULL)
            throw std::runtime_error("The safetensors header is invalid.");
        std::string header(static_cast<std::size_t>(headerSize), '\0');
        stream.read(header.data(), static_cast<std::streamsize>(header.size()));
        if (!stream)
            throw std::runtime_error("Unable to read the safetensors header.");

        GPT model(ConfigFor(type));
        const auto dataStart = static_cast<std::streamoff>(sizeof(headerSize) + headerSize);
        const std::regex entry(R"safetensors("([^"]+)"\s*:\s*\{\s*"dtype"\s*:\s*"([^"]+)"\s*,\s*"shape"\s*:\s*\[([^\]]*)\]\s*,\s*"data_offsets"\s*:\s*\[([0-9]+)\s*,\s*([0-9]+)\]\s*\})safetensors");
        for (std::sregex_iterator match(header.begin(), header.end(), entry), end; match != end; ++match)
        {
            const auto sourceName = (*match)[1].str();
            if ((*match)[2].str() != "F32")
                throw std::runtime_error("Only F32 safetensors checkpoints are supported: " + sourceName);
            auto shape = ParseShape((*match)[3].str());
            std::size_t elements = 1;
            for (const auto dimension : shape)
            {
                if (dimension <= 0 || elements > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension))
                    throw std::runtime_error("Invalid tensor shape: " + sourceName);
                elements *= static_cast<std::size_t>(dimension);
            }
            const auto offset = std::stoull((*match)[4].str());
            const auto endOffset = std::stoull((*match)[5].str());
            if (endOffset < offset || endOffset - offset != elements * sizeof(float))
                throw std::runtime_error("Invalid data range for tensor: " + sourceName);
            std::vector<float> values(elements);
            stream.seekg(dataStart + static_cast<std::streamoff>(offset));
            stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
            if (!stream)
                throw std::runtime_error("Unable to read tensor data: " + sourceName);
            auto name = sourceName.rfind("transformer.", 0) == 0 ? sourceName.substr(12) : sourceName;
            const auto isAttentionMask = name.size() >= 10 && name.compare(name.size() - 10, 10, ".attn.bias") == 0;
            const auto isMaskedBias = name.size() >= 17 && name.compare(name.size() - 17, 17, ".attn.masked_bias") == 0;
            if (name == "lm_head.weight" || isAttentionMask || isMaskedBias)
                continue;
            Tensor tensor(std::move(values), std::move(shape));
            if (IsTransposedWeight(name))
                tensor = TransposeWeight(tensor);
            model.implementation_->parameters.insert_or_assign(std::move(name), std::move(tensor));
        }
        auto tokenEmbedding = model.implementation_->parameters.find("wte.weight");
        if (tokenEmbedding == model.implementation_->parameters.end())
            throw std::runtime_error("The checkpoint does not contain transformer.wte.weight.");
        return model;
    }

    void GPT2Tokenizer::load(const std::string& encoderJsonPath, const std::string& vocabBpePath)
    {
        std::ifstream encoderFile(encoderJsonPath);
        std::ifstream vocabFile(vocabBpePath);
        if (!encoderFile || !vocabFile) throw std::runtime_error("Unable to open GPT-2 tokenizer assets.");
        const std::string encoder((std::istreambuf_iterator<char>(encoderFile)), {});
        encoder_.clear();
        decoder_.clear();
        const std::regex entry(R"json("((\\.|[^"\\])*)"\s*:\s*([0-9]+))json");
        for (std::sregex_iterator match(encoder.begin(), encoder.end(), entry), end; match != end; ++match)
        {
            const auto token = JsonUnescape((*match)[1].str());
            const auto id = std::stoi((*match)[3].str());
            encoder_.insert_or_assign(token, id);
            decoder_.insert_or_assign(id, token);
        }
        if (encoder_.size() != 50257 || decoder_.size() != 50257)
            throw std::runtime_error("The GPT-2 encoder vocabulary is incomplete.");
        std::string line;
        int rank = 0;
        while (std::getline(vocabFile, line))
            if (!line.empty() && line[0] != '#') mergeRanks_.emplace(line, rank++);
    }

    std::vector<int> GPT2Tokenizer::encode(const std::string& text) const
    {
        if (encoder_.empty()) throw std::logic_error("Load tokenizer assets before encoding.");
        std::vector<int> result;
        for (std::size_t start = 0; start < text.size(); )
        {
            auto end = start;
            if (std::isspace(static_cast<unsigned char>(text[start])))
            {
                ++end;
                while (end < text.size() && !std::isspace(static_cast<unsigned char>(text[end]))) ++end;
            }
            else
            {
                ++end;
                while (end < text.size() && !std::isspace(static_cast<unsigned char>(text[end]))) ++end;
            }

            std::string encoded;
            for (std::size_t index = start; index < end; ++index)
                encoded += ByteEncoder()[static_cast<unsigned char>(text[index])];
            const auto pieces = Bpe(encoded, mergeRanks_);
            for (const auto& piece : pieces)
            {
                const auto iterator = encoder_.find(piece);
                if (iterator == encoder_.end()) throw std::runtime_error("Text cannot be represented by the loaded GPT-2 vocabulary.");
                result.push_back(iterator->second);
            }
            start = end;
        }
        return result;
    }

    std::string GPT2Tokenizer::decode(const std::vector<int>& tokens) const
    {
        std::string encoded;
        for (const auto token : tokens)
        {
            const auto iterator = decoder_.find(token);
            if (iterator == decoder_.end()) throw std::out_of_range("Token ID is not in the loaded vocabulary.");
            encoded += iterator->second;
        }

        std::string result;
        const auto& decoder = ByteDecoder();
        for (const auto& piece : Utf8Pieces(encoded))
        {
            const auto iterator = decoder.find(piece);
            if (iterator == decoder.end())
                throw std::runtime_error("The GPT-2 vocabulary contains an invalid byte token.");
            result.push_back(static_cast<char>(iterator->second));
        }
        return result;
    }

    GPT2Service::GPT2Service(GPT model, GPT2Tokenizer tokenizer) : model_(std::move(model)), tokenizer_(std::move(tokenizer)) {}

    std::string GPT2Service::generateText(const std::string& input, const GenerationSettings& settings) const
    {
		if (settings.maxTokens <= 0 || settings.temperature <= 0.0F || settings.topK <= 0 || settings.ngramSize <= 0)
            throw std::invalid_argument("Generation settings must be positive.");
        auto tokens = tokenizer_.encode(input);
        if (tokens.empty()) throw std::invalid_argument("The prompt must produce at least one token.");
        std::mt19937 random(settings.seed);
        for (int generated = 0; generated < settings.maxTokens && static_cast<int>(tokens.size()) < model_.config().blockSize; ++generated)
        {
            std::vector<float> inputValues;
            inputValues.reserve(tokens.size());
            for (const auto token : tokens)
                inputValues.push_back(static_cast<float>(token));
            const auto logits = model_.forward(Tensor(std::move(inputValues), { 1, static_cast<int>(tokens.size()) }));
            std::vector<float> scores(model_.config().vocabSize);
            const auto sourceOffset = (static_cast<int>(tokens.size()) - 1) * model_.config().vocabSize;
            std::copy_n(logits.values().begin() + sourceOffset, scores.size(), scores.begin());
            std::set<int> used(tokens.begin(), tokens.end());
            for (const auto token : used) scores[token] = scores[token] > 0 ? scores[token] / settings.repetitionPenalty : scores[token] * settings.repetitionPenalty;
            if (static_cast<int>(tokens.size()) >= settings.ngramSize)
                for (int start = 0; start + settings.ngramSize <= static_cast<int>(tokens.size()); ++start)
                    if (std::equal(tokens.begin() + start, tokens.begin() + start + settings.ngramSize - 1, tokens.end() - (settings.ngramSize - 1))) scores[tokens[start + settings.ngramSize - 1]] = -std::numeric_limits<float>::infinity();
            for (auto& score : scores) score /= settings.temperature;
            std::vector<int> indices(scores.size()); std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + std::min(settings.topK, static_cast<int>(indices.size())), indices.end(), [&scores](int left, int right) { return scores[left] > scores[right]; });
            const auto allowed = std::min(settings.topK, static_cast<int>(indices.size()));
            for (int index = allowed; index < static_cast<int>(indices.size()); ++index) scores[indices[index]] = -std::numeric_limits<float>::infinity();
            const auto probabilities = softmax(Tensor(scores, { 1, static_cast<int>(scores.size()) }));
            std::discrete_distribution<int> distribution(probabilities.values().begin(), probabilities.values().end());
            tokens.push_back(distribution(random));
            std::clog << "Generated token " << generated + 1 << " of " << settings.maxTokens
                      << " (total tokens: " << tokens.size() << ")\n";
            const auto recent = tokenizer_.decode(std::vector<int>(tokens.end() - std::min<std::size_t>(tokens.size(), 10), tokens.end()));
            if (recent.find("\nQ:") != std::string::npos || recent.find("\nUser") != std::string::npos || recent.find("<|endoftext|>") != std::string::npos) break;
        }
        return tokenizer_.decode(tokens);
    }
}

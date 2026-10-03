#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace gptplus
{
	class Tensor
	{
	public:
		Tensor() = default;
		explicit Tensor(std::vector<int> shape);
		Tensor(std::vector<float> values, std::vector<int> shape);

		const std::vector<int>& shape() const noexcept;
		const std::vector<int>& strides() const noexcept;
		const std::vector<float>& values() const noexcept;
		std::vector<float>& values() noexcept;
		int rank() const noexcept;
		int length() const noexcept;
		int size(int dimension) const;
		float at(const std::vector<int>& indices) const;
		float& at(const std::vector<int>& indices);
		Tensor reshape(std::vector<int> newShape) const;
		Tensor transpose(const std::vector<int>& dimensions) const;
		Tensor transpose(int firstDimension, int secondDimension) const;
		Tensor slice(int dimension, int start, int length) const;
		Tensor repeat(const std::vector<int>& repeats) const;
		Tensor add(const Tensor& other) const;
		Tensor add(float scalar) const;
		Tensor multiply(float scalar) const;
		Tensor divide(float scalar) const;
		Tensor power(float exponent) const;
		Tensor tanh() const;

		static Tensor zeros(std::vector<int> shape);
		static Tensor ones(std::vector<int> shape);

	private:
		std::vector<float> values_;
		std::vector<int> shape_;
		std::vector<int> strides_;
		int flatIndex(const std::vector<int>& indices) const;
	};

	struct GPTConfig
	{
		int blockSize = 1024;
		int vocabSize = 50257;
		int layerCount = 12;
		int headCount = 12;
		int embeddingSize = 768;
	};

	enum class GPT2ModelType { GPT2, GPT2Medium, GPT2Large, GPT2XL };

	struct GenerationSettings
	{
		float temperature = 0.5F;
		int topK = 40;
		float topP = 0.7F;
		float repetitionPenalty = 1.2F;
		int ngramSize = 3;
		int maxTokens = 60;
		std::uint32_t seed = 1337;
	};

	class GPT2Tokenizer
	{
	public:
		void load(const std::string& encoderJsonPath, const std::string& vocabBpePath);
		std::vector<int> encode(const std::string& text) const;
		std::string decode(const std::vector<int>& tokens) const;

	private:
		std::unordered_map<std::string, int> encoder_;
		std::unordered_map<int, std::string> decoder_;
		std::unordered_map<std::string, int> mergeRanks_;
	};

	class GPT
	{
	public:
		struct Implementation;
		struct Cache
		{
			int tokenCount = 0;
			std::vector<std::vector<float>> keys;
			std::vector<std::vector<float>> values;
		};

		explicit GPT(GPTConfig config);
		static GPT load(const std::string& safetensorsPath, GPT2ModelType type = GPT2ModelType::GPT2);
		Tensor forward(const Tensor& tokenIds) const;
		Cache createCache() const;
		Tensor forwardCached(const Tensor& tokenIds, Cache& cache) const;
		const GPTConfig& config() const noexcept;

	private:
		std::shared_ptr<Implementation> implementation_;
	};

	class GPT2Service
	{
	public:
		GPT2Service(GPT model, GPT2Tokenizer tokenizer);
		std::string generateText(const std::string& input, const GenerationSettings& settings = {}) const;

	private:
		GPT model_;
		GPT2Tokenizer tokenizer_;
	};

	Tensor softmax(const Tensor& input);
	Tensor concatenateLastDimension(const Tensor& left, const Tensor& right);
}

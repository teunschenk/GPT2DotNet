#include "GPTPlusLib.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace gptplus
{
	namespace
	{
		int ElementCount(const std::vector<int>& shape)
		{
			if (shape.empty()) throw std::invalid_argument("A tensor requires at least one dimension.");
			int result = 1;
			for (const auto dimension : shape)
			{
				if (dimension <= 0 || result > INT_MAX / dimension) throw std::invalid_argument("Invalid tensor shape.");
				result *= dimension;
			}
			return result;
		}

		std::vector<int> CreateStrides(const std::vector<int>& shape)
		{
			std::vector<int> result(shape.size());
			int stride = 1;
			for (auto index = static_cast<int>(shape.size()) - 1; index >= 0; --index)
			{
				result[index] = stride;
				stride *= shape[index];
			}
			return result;
		}

		int NormalizeDimension(int dimension, int rank)
		{
			if (dimension < 0) dimension += rank;
			if (dimension < 0 || dimension >= rank) throw std::out_of_range("Tensor dimension is out of range.");
			return dimension;
		}

		std::vector<int> Indices(int flatIndex, const std::vector<int>& shape, const std::vector<int>& strides)
		{
			std::vector<int> result(shape.size());
			for (std::size_t axis = 0; axis < shape.size(); ++axis)
			{
				result[axis] = flatIndex / strides[axis];
				flatIndex %= strides[axis];
			}
			return result;
		}
	}

	Tensor::Tensor(std::vector<int> shape) : values_(ElementCount(shape)), shape_(std::move(shape)), strides_(CreateStrides(shape_)) {}
	Tensor::Tensor(std::vector<float> values, std::vector<int> shape) : values_(std::move(values)), shape_(std::move(shape)), strides_(CreateStrides(shape_))
	{
		if (static_cast<int>(values_.size()) != ElementCount(shape_)) throw std::invalid_argument("The number of values must match the tensor shape.");
	}

	const std::vector<int>& Tensor::shape() const noexcept { return shape_; }
	const std::vector<int>& Tensor::strides() const noexcept { return strides_; }
	const std::vector<float>& Tensor::values() const noexcept { return values_; }
	std::vector<float>& Tensor::values() noexcept { return values_; }
	int Tensor::rank() const noexcept { return static_cast<int>(shape_.size()); }
	int Tensor::length() const noexcept { return static_cast<int>(values_.size()); }
	int Tensor::size(int dimension) const { return shape_[NormalizeDimension(dimension, rank())]; }

	int Tensor::flatIndex(const std::vector<int>& indices) const
	{
		if (indices.size() != shape_.size()) throw std::invalid_argument("An index is required for every tensor dimension.");
		int result = 0;
		for (std::size_t axis = 0; axis < indices.size(); ++axis)
		{
			if (indices[axis] < 0 || indices[axis] >= shape_[axis]) throw std::out_of_range("Tensor index is out of range.");
			result += indices[axis] * strides_[axis];
		}
		return result;
	}

	float Tensor::at(const std::vector<int>& indices) const { return values_[flatIndex(indices)]; }
	float& Tensor::at(const std::vector<int>& indices) { return values_[flatIndex(indices)]; }
	Tensor Tensor::zeros(std::vector<int> shape) { return Tensor(std::move(shape)); }
	Tensor Tensor::ones(std::vector<int> shape) { Tensor result(std::move(shape)); std::fill(result.values_.begin(), result.values_.end(), 1.0F); return result; }
	Tensor Tensor::reshape(std::vector<int> newShape) const { if (ElementCount(newShape) != length()) throw std::invalid_argument("Reshape must preserve element count."); return Tensor(values_, std::move(newShape)); }

	Tensor Tensor::transpose(const std::vector<int>& dimensions) const
	{
		if (static_cast<int>(dimensions.size()) != rank()) throw std::invalid_argument("A dimension must be supplied for every tensor axis.");
		std::vector<bool> seen(rank());
		std::vector<int> outputShape(rank());
		for (int axis = 0; axis < rank(); ++axis)
		{
			const auto dimension = dimensions[axis];
			if (dimension < 0 || dimension >= rank() || seen[dimension]) throw std::invalid_argument("Dimensions must be a permutation of tensor axes.");
			seen[dimension] = true;
			outputShape[axis] = shape_[dimension];
		}
		Tensor output(outputShape);
		for (int flat = 0; flat < output.length(); ++flat)
		{
			const auto outputIndices = Indices(flat, output.shape_, output.strides_);
			std::vector<int> sourceIndices(rank());
			for (int axis = 0; axis < rank(); ++axis) sourceIndices[dimensions[axis]] = outputIndices[axis];
			output.values_[flat] = at(sourceIndices);
		}
		return output;
	}

	Tensor Tensor::transpose(int firstDimension, int secondDimension) const
	{
		firstDimension = NormalizeDimension(firstDimension, rank());
		secondDimension = NormalizeDimension(secondDimension, rank());
		std::vector<int> dimensions(rank());
		std::iota(dimensions.begin(), dimensions.end(), 0);
		std::swap(dimensions[firstDimension], dimensions[secondDimension]);
		return transpose(dimensions);
	}

	Tensor Tensor::slice(int dimension, int start, int count) const
	{
		dimension = NormalizeDimension(dimension, rank());
		if (start < 0 || count <= 0 || start > shape_[dimension] - count) throw std::out_of_range("Tensor slice is out of range.");
		auto outputShape = shape_; outputShape[dimension] = count;
		Tensor output(outputShape);
		for (int flat = 0; flat < output.length(); ++flat)
		{
			auto indices = Indices(flat, output.shape_, output.strides_);
			indices[dimension] += start;
			output.values_[flat] = at(indices);
		}
		return output;
	}

	Tensor Tensor::repeat(const std::vector<int>& repeats) const
	{
		if (repeats.size() != shape_.size()) throw std::invalid_argument("A repeat count is required for every tensor dimension.");
		std::vector<int> outputShape(rank());
		for (int axis = 0; axis < rank(); ++axis)
		{
			if (repeats[axis] <= 0) throw std::invalid_argument("Repeat counts must be positive.");
			outputShape[axis] = shape_[axis] * repeats[axis];
		}
		Tensor output(outputShape);
		for (int flat = 0; flat < output.length(); ++flat)
		{
			auto indices = Indices(flat, output.shape_, output.strides_);
			for (int axis = 0; axis < rank(); ++axis) indices[axis] %= shape_[axis];
			output.values_[flat] = at(indices);
		}
		return output;
	}

	Tensor Tensor::add(const Tensor& other) const
	{
		const auto outputRank = std::max(rank(), other.rank());
		std::vector<int> outputShape(outputRank);
		for (int axis = 0; axis < outputRank; ++axis)
		{
			const auto leftAxis = axis - (outputRank - rank());
			const auto rightAxis = axis - (outputRank - other.rank());
			const auto leftSize = leftAxis < 0 ? 1 : shape_[leftAxis];
			const auto rightSize = rightAxis < 0 ? 1 : other.shape_[rightAxis];
			if (leftSize != rightSize && leftSize != 1 && rightSize != 1) throw std::invalid_argument("Tensor shapes are not broadcast-compatible.");
			outputShape[axis] = std::max(leftSize, rightSize);
		}
		Tensor output(outputShape);
		for (int flat = 0; flat < output.length(); ++flat)
		{
			const auto outputIndices = Indices(flat, output.shape_, output.strides_);
			std::vector<int> leftIndices(rank()), rightIndices(other.rank());
			for (int axis = 0; axis < outputRank; ++axis)
			{
				const auto leftAxis = axis - (outputRank - rank());
				const auto rightAxis = axis - (outputRank - other.rank());
				if (leftAxis >= 0) leftIndices[leftAxis] = shape_[leftAxis] == 1 ? 0 : outputIndices[axis];
				if (rightAxis >= 0) rightIndices[rightAxis] = other.shape_[rightAxis] == 1 ? 0 : outputIndices[axis];
			}
			output.values_[flat] = at(leftIndices) + other.at(rightIndices);
		}
		return output;
	}

	Tensor Tensor::add(float scalar) const { auto output = *this; for (auto& value : output.values_) value += scalar; return output; }
	Tensor Tensor::multiply(float scalar) const { auto output = *this; for (auto& value : output.values_) value *= scalar; return output; }
	Tensor Tensor::divide(float scalar) const { if (scalar == 0.0F) throw std::invalid_argument("Cannot divide by zero."); return multiply(1.0F / scalar); }
	Tensor Tensor::power(float exponent) const { auto output = *this; for (auto& value : output.values_) value = std::pow(value, exponent); return output; }
	Tensor Tensor::tanh() const { auto output = *this; for (auto& value : output.values_) value = std::tanh(value); return output; }
}

namespace gptplus
{
    Tensor softmax(const Tensor& input)
    {
        if (input.rank() == 0) throw std::invalid_argument("Softmax requires a ranked tensor.");
        Tensor output(input.shape());
        const auto width = input.shape().back();
        for (int offset = 0; offset < input.length(); offset += width)
        {
            const auto maximum = *std::max_element(input.values().begin() + offset, input.values().begin() + offset + width);
            float sum = 0.0F;
            for (int index = 0; index < width; ++index)
            {
                const auto value = std::exp(input.values()[offset + index] - maximum);
                output.values()[offset + index] = value;
                sum += value;
            }
            for (int index = 0; index < width; ++index) output.values()[offset + index] /= sum;
        }
        return output;
    }

    Tensor concatenateLastDimension(const Tensor& left, const Tensor& right)
    {
        if (left.rank() != right.rank() || left.rank() == 0) throw std::invalid_argument("Concatenation requires matching non-zero ranks.");
        for (int axis = 0; axis < left.rank() - 1; ++axis)
            if (left.shape()[axis] != right.shape()[axis]) throw std::invalid_argument("Tensor shapes must match outside the final dimension.");
        auto outputShape = left.shape();
        outputShape.back() += right.shape().back();
        Tensor output(outputShape);
        const auto groups = output.length() / outputShape.back();
        for (int group = 0; group < groups; ++group)
        {
            const auto outputOffset = group * outputShape.back();
            std::copy_n(left.values().begin() + group * left.shape().back(), left.shape().back(), output.values().begin() + outputOffset);
            std::copy_n(right.values().begin() + group * right.shape().back(), right.shape().back(), output.values().begin() + outputOffset + left.shape().back());
        }
        return output;
    }
}

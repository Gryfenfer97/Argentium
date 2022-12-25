#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace Ag{

	class Neuron{
	private:
		std::vector<float> lastInput;
		std::vector<float> weights;
		float bias;

		

	public:
		Neuron() = default;
		explicit Neuron(const std::size_t nbInput);

		float feedForward(const std::vector<float>& inputs);
		std::vector<float> backProp(const float cost);
	};

}
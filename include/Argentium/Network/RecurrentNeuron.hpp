#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace Ag{

	class RecurrentNeuron{
	private:
		float z;
		std::vector<float> lastInput;
		std::vector<float> weights;
        float lastOutput;
		float bias;

	public:
		RecurrentNeuron() = default;
		explicit RecurrentNeuron(const std::size_t nbInput);

		float feedForward(const std::vector<float>& inputs);
		std::vector<float> backProp(const float cost);
	};

}


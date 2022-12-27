#include <Argentium/Network/RecurrentNeuron.hpp>

using namespace Ag;

#define ALPHA 0.01f

RecurrentNeuron::RecurrentNeuron(const std::size_t nbInput)
{
	z = 0;
	bias = 1.f;
	this->weights.resize(nbInput + 1, 0.f);
	const float valueMax = 2.4f / sqrtf(static_cast<float>(nbInput + 1));

	std::generate(this->weights.begin(), this->weights.end(), [valueMax]
				  { return 2 * valueMax * std::rand() / static_cast<float>(RAND_MAX) - valueMax; });
}

float RecurrentNeuron::feedForward(const std::vector<float> &inputs)
{
	this->lastInput = inputs;
	this->lastOutput = std::inner_product(
		inputs.begin(),
		inputs.end(),
		this->weights.begin(),
		this->bias + this->lastOutput * this->weights.back());
	return this->lastOutput;
}

std::vector<float> RecurrentNeuron::backProp(const float cost)
{
	std::vector<float> costs(this->weights.size() - 1);

	std::transform(this->weights.begin(), std::next(this->weights.end() - 1), costs.begin(), [&cost](const float weight) -> float
				   { return cost * weight; });

	// Update Weights
	std::transform(this->weights.cbegin(), this->weights.cend(), this->lastInput.cbegin(), this->weights.begin(),
				   [cost](const auto &weight, const auto &input)
				   {
					   const float deltaWeight = cost * input;
					   return weight + deltaWeight * ALPHA;
				   });

	return costs;
}
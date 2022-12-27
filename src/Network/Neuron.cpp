#include <Argentium/Network/Neuron.hpp>

#define ALPHA 0.01f

using namespace Ag;

Neuron::Neuron(const std::size_t nbInput)
{
	bias = 1.f;
	this->weights.resize(nbInput, 0.f);
	const float valueMax = 2.4f / sqrtf(static_cast<float>(nbInput));

	std::generate(this->weights.begin(), this->weights.end(), [valueMax]
				  { return 2 * valueMax * std::rand() / static_cast<float>(RAND_MAX) - valueMax; });
}

float Neuron::feedForward(const std::vector<float> &inputs)
{
	this->lastInput = inputs;
	return std::inner_product(
		inputs.begin(),
		inputs.end(),
		this->weights.begin(),
		this->bias
	);
}

std::vector<float> Neuron::backProp(const float cost)
{
	std::vector<float> costs(this->weights.size());

	std::transform(this->weights.begin(), this->weights.end(), costs.begin(), [&cost](const float weight) -> float
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

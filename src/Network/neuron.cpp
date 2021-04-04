#include <Argentium/Network/neuron.hpp>

#define ALPHA 0.01f

using namespace Ag;

Neuron::Neuron(const std::size_t nbInput)
{
	z = 0;
	bias = 1.f;
	this->weights.resize(nbInput, 0.f);
	const float valueMax = 2.4f / sqrtf(static_cast<float>(nbInput));
	
	for(auto& weight : this->weights)
	{
		weight = rand() / static_cast<float>(RAND_MAX) * 2 * valueMax - valueMax;
	}
	
}

float Neuron::feedForward(const std::vector<float>& inputs)
{
	std::vector<float> mul(inputs.size());
	this->lastInput = inputs;
	std::transform(inputs.begin(), inputs.end(), this->weights.begin(), mul.begin(), std::multiplies<>{});
	return std::accumulate(mul.begin(), mul.end(), 0.f) + this->bias;
}

std::vector<float> Neuron::backProp(const float cost)
{
	std::vector<float> costs;
	costs.resize(this->weights.size());

	std::transform(this->weights.begin(), this->weights.end(), costs.begin(), [&cost](const float weight)->float {return cost * weight; });

	//Update Weights
	for(std::size_t i = 0;i < this->weights.size();i++)
	{
		const float deltaWeight = cost * lastInput[i];
		this->weights[i] += deltaWeight * ALPHA;
	}
	
	return costs;
}

#include <Argentium/Network/FullyConnected.hpp>
using namespace Ag;

FullyConnected::FullyConnected(const std::size_t nbNeurons, const std::size_t nbInputs, Internal::ActivationFunctions activations)
{
	this->numberOfInput = nbInputs;
	this->activationFunctions = activations;
	this->neurons.reserve(nbNeurons);
	for (std::size_t i = 0; i < nbNeurons; i++)
	{
		this->neurons.push_back(Neuron(nbInputs));
	}
	this->lastOutput.resize(nbNeurons);
}

std::vector<float> FullyConnected::feedForward(const std::vector<float> &inputs)
{

	std::transform(this->neurons.begin(), this->neurons.end(), this->lastOutput.begin(), [inputs](auto &neuron)
				   { return neuron.feedForward(inputs); });
	return std::get<0>(this->activationFunctions)(this->lastOutput);
}

std::vector<float> FullyConnected::backProp(const std::vector<float> &inputCosts)
{
	auto newInputCosts = inputCosts;
	const auto r = std::get<1>(this->activationFunctions)(this->lastOutput);
	std::transform(newInputCosts.begin(), newInputCosts.end(), r.begin(), newInputCosts.begin(), std::multiplies<>());

	std::vector<float> costs(this->numberOfInput, 0);
	for (std::size_t n = 0; n < this->neurons.size(); n++)
	{
		std::vector<float> costsForOneNeuron = this->neurons.at(n).backProp(newInputCosts.at(n));
		std::transform(costs.begin(), costs.end(), costsForOneNeuron.begin(), costs.begin(), std::plus<>());
	}
	return costs;
}

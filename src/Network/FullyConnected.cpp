#include <Argentium/Network/FullyConnected.hpp>

using namespace Ag;


FullyConnected::FullyConnected(const std::size_t nbNeurons, const std::size_t nbInputs, Internal::ActivationFunctions activations)
{
	this->numberOfInput = nbInputs;
	this->activationFunctions = activations;
	this->neurons.resize(nbNeurons);
	for(std::size_t i = 0;i<nbNeurons;i++)
	{
		this->neurons[i] = Neuron(nbInputs);
	}
}

std::vector<float> FullyConnected::feedForward(const std::vector<float>& inputs)
{
	std::vector<float> res(this->neurons.size());
	
	for(std::size_t i=0; i < this->neurons.size();i++)
	{
		res[i] = this->neurons[i].feedForward(inputs);
	}

	lastOutput = res;

	res = std::get<0>(this->activationFunctions)(res);

    return res;
}

std::vector<float> FullyConnected::backProp(const std::vector<float>& inputCosts)
{
	auto newInputCosts = inputCosts;
	const auto activationFunctionDerivate = std::get<1>(this->activationFunctions);
	auto r = activationFunctionDerivate(lastOutput);
	std::transform(newInputCosts.begin(), newInputCosts.end(), r.begin(), newInputCosts.begin(), std::multiplies<>());


	std::vector<float> costs(this->numberOfInput, 0);
	for (std::size_t n = 0; n < this->neurons.size(); n++) {
		std::vector<float> costsForOneNeuron = this->neurons[n].backProp(newInputCosts[n]);
		std::transform(costs.begin(), costs.end(), costsForOneNeuron.begin(), costs.begin(), std::plus<>());
	}
	return costs;
}

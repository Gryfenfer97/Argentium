#include <Argentium/Network/Recurrence.hpp>

using namespace Ag;

Recurrence::Recurrence(const std::size_t nbNeurons, const std::size_t nbInputs, Internal::ActivationFunctions activations)
{
    this->numberOfInput = nbInputs;
    this->activationFunctions = activations;
    this->neurons.resize(nbNeurons);
    for (std::size_t i = 0; i < nbNeurons; i++)
    {
        this->neurons[i] = RecurrentNeuron(nbInputs);
    }
}

std::vector<float> Recurrence::feedForward(const std::vector<float> &inputs)
{
    std::vector<float> res(this->neurons.size());

    std::transform(this->neurons.begin(), this->neurons.end(), res.begin(), [inputs](auto &neuron)
                   { return neuron.feedForward(inputs); });

    lastOutput = res;

    return std::get<0>(this->activationFunctions)(res);
}

std::vector<float> Recurrence::backProp(const std::vector<float> &inputCosts)
{
    auto newInputCosts = inputCosts;
    const auto activationFunctionDerivate = std::get<1>(this->activationFunctions);
    auto r = activationFunctionDerivate(lastOutput);
    std::transform(newInputCosts.begin(), newInputCosts.end(), r.begin(), newInputCosts.begin(), std::multiplies<>());

    std::vector<float> costs(this->numberOfInput, 0);
    for (std::size_t n = 0; n < this->neurons.size(); n++)
    {
        std::vector<float> costsForOneNeuron = this->neurons[n].backProp(newInputCosts[n]);
        std::transform(costs.begin(), costs.end(), costsForOneNeuron.begin(), costs.begin(), std::plus<>());
    }
    return costs;
}

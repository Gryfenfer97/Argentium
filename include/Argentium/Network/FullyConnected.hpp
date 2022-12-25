#pragma once
#include <Argentium/Network/BaseLayer.hpp>
#include <Argentium/Network/neuron.hpp>


namespace Ag
{
	class FullyConnected final : public BaseLayer
	{
	private:
		std::vector<Neuron> neurons;
		std::vector<float> lastOutput;
	
	public:

		FullyConnected() = default;
		FullyConnected(const std::size_t nbNeurons, const std::size_t nbInputs, Internal::ActivationFunctions activations);

		std::vector<float> feedForward(const std::vector<float>& inputs) override;
		std::vector<float> backProp(const std::vector<float>& inputCosts) override;
	};

	struct FullyConnectedBuilder : public LayerBuilder
	{
		std::unique_ptr<BaseLayer> createLayer(std::size_t numberOfInput) override
		{
			return std::make_unique<FullyConnected>(this->numberOfOutput, numberOfInput, this->activationFunctions);
		}

		FullyConnectedBuilder(const std::size_t numberOfOutput, const Internal::ActivationFunctions& activationFunctions)
		{
			this->numberOfOutput = numberOfOutput;
			this->activationFunctions = activationFunctions;
		}
	};

	inline std::shared_ptr<FullyConnectedBuilder> FullyConnectedLayer(const std::size_t numberOfNeuron, const Internal::ActivationFunctions& activationFunctions)
	{
		return std::make_shared<FullyConnectedBuilder>(numberOfNeuron, activationFunctions);
	}
}

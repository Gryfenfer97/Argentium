#pragma once
#include <Argentium/Network/BaseLayer.hpp>
#include <Argentium/Network/RecurrentNeuron.hpp>

namespace Ag{
    class Recurrence final : public BaseLayer{
    private:
        std::vector<RecurrentNeuron> neurons;
        std::vector<float> lastOutput;

    public:
        Recurrence() = default;
        Recurrence(const std::size_t nbNeurons, const std::size_t nbInputs, Internal::ActivationFunctions activations);

        std::vector<float> feedForward(const std::vector<float>& inputs) override;
		std::vector<float> backProp(const std::vector<float>& inputCosts) override;
    };

    struct RecurenceBuilder : public LayerBuilder
	{
		std::unique_ptr<BaseLayer> createLayer(std::size_t numberOfInput) override
		{
			return std::make_unique<Recurrence>(this->numberOfOutput, numberOfInput, this->activationFunctions);
		}

		RecurenceBuilder(const std::size_t numberOfOutput, const Internal::ActivationFunctions& activationFunctions)
		{
			this->numberOfOutput = numberOfOutput;
			this->activationFunctions = activationFunctions;
		}
	};

	inline std::shared_ptr<RecurenceBuilder> RecurrenceLayer(const std::size_t numberOfNeuron, const Internal::ActivationFunctions& activationFunctions)
	{
		return std::make_shared<RecurenceBuilder>(numberOfNeuron, activationFunctions);
	}
}
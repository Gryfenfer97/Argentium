#pragma once
#include <Argentium/Network/BaseLayer.hpp>

namespace Ag
{

	class Input : public BaseLayer
	{
	public:
		Input() = default;
		Input(const std::size_t numberOfInput)
		{
			this->numberOfInput = numberOfInput;
		};

		std::vector<float> feedForward(const std::vector<float>& inputs) override { return std::vector<float>(); };
		std::vector<float> backProp(const std::vector<float>& inputCosts) override { return std::vector<float>(); };
	};

	
	struct InputFactory : public LayerFactory
	{
		std::unique_ptr<BaseLayer> createLayer(std::size_t numberOfInput) override
		{
			return std::make_unique<Input>(numberOfInput);
		}

		InputFactory(const std::size_t numberOfOutput)
		{
			this->numberOfOutput = numberOfOutput;
			this->activationFunctions = activationFunctions;
		}
	};

	inline std::shared_ptr<InputFactory> InputLayer(const std::size_t numberOfInput)
	{
		return std::make_shared<InputFactory>(numberOfInput);
	}
	
}
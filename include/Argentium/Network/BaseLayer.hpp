#pragma once
#include <vector>
#include <functional>
#include <memory>

namespace Ag
{
	namespace Internal
	{
		typedef std::function<std::vector<float>(std::vector<float>)> ActivationFunction;
		typedef std::tuple< ActivationFunction, ActivationFunction> ActivationFunctions;
	}

	
	class BaseLayer
	{
	protected:
		std::size_t numberOfInput;
		Internal::ActivationFunctions activationFunctions;

	public:
		BaseLayer() = default;
		virtual ~BaseLayer() = default;


		virtual std::vector<float> feedForward(const std::vector<float>& inputs) = 0;
		virtual std::vector<float> backProp(const std::vector<float>& inputCosts) = 0;
	};

	struct LayerBuilder
	{
	public:
		std::size_t numberOfOutput;
		Internal::ActivationFunctions activationFunctions;

	public:
		virtual std::unique_ptr<BaseLayer> createLayer([[maybe_unused]] std::size_t outputNumber) { return nullptr; }
	};
	
}

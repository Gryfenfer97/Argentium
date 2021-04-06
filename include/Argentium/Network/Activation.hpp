#pragma once
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

namespace Ag{
	namespace Activation
		{
		namespace Internal
		{
			typedef std::function<std::vector<float>(std::vector<float>)> ActivationFunction;
			typedef std::tuple< ActivationFunction, ActivationFunction> ActivationFunctions;


			static std::vector<float> identityFunction(const std::vector<float>& inputs)
			{
				return inputs;
			}

			static std::vector<float> identityDerivate(const std::vector<float>& inputs)
			{
				return std::vector<float>(inputs.size(), 1);
			}

			static std::vector<float> sigmoidFunction(const std::vector<float>& inputs)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), [](const float& z)->float {return 1.0f / (1.0f + std::exp(-z)); });
				return res;
			}

			static std::vector<float> sigmoidDerivate(const std::vector<float>& inputs)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), [](const float& z)->float {return (std::exp(-z) / powf((std::exp(-z) + 1.0f), 2)); });
				return res;
			}

			static std::vector<float> softmaxFunction(const std::vector<float>& inputs)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), [](const float& z)->float {return std::exp(z); });
				auto sum = std::accumulate(res.begin(), res.end(), 0.f);
				std::transform(res.begin(), res.end(), res.begin(), [&sum](const float& z)->float {return z / sum; });
				return res;
			}

			static std::vector<float> softmaxDerivate(const std::vector<float>& inputs)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), [](const float& z)->float {return std::exp(z); });
				auto sum = std::accumulate(res.begin(), res.end(), 0.f);
				std::transform(res.begin(), res.end(), res.begin(), [&sum](const float& z)->float {return z * (sum - 1) / (sum * sum); });
				return res;
			}

			static std::vector<float> tanhFunction(const std::vector<float>& inputs)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), std::tanhf);
				return res;
			}

			static std::vector<float> tanhDerivate(const std::vector<float>& inputs)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), [](const float& z)->float {return 1 - std::tanhf(z) * std::tanhf(z); });
				return res;
			}

			static std::vector<float> ReLUFunction(const std::vector<float>& inputs)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), [](const float& z)->float {return std::max(0.f, z); });
				return res;
			}

			static std::vector<float> ReLUDerivate(const std::vector<float>& inputs)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), [](const float& z)->float {return z > 0 ? 1.f : 0.f; }); //heavyside function
				return res;
			}

			static std::vector<float> leakyReLUFunction(const std::vector<float>& inputs, const float alpha)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), [&alpha](const float& z)->float {return z > 0 ? z : z * alpha; });
				return res;
			}

			static std::vector<float> leakyReLUDerivate(const std::vector<float>& inputs, const float alpha)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), [&alpha](const float& z)->float {return z > 0 ? 1.f : alpha; });
				return res;
			}
		}
		

			static Internal::ActivationFunctions sigmoid = std::make_tuple(Internal::sigmoidFunction, Internal::sigmoidDerivate);
			static Internal::ActivationFunctions softmax = std::make_tuple(Internal::softmaxFunction, Internal::sigmoidDerivate);
			static Internal::ActivationFunctions identity = std::make_tuple(Internal::identityFunction, Internal::identityDerivate);
			static Internal::ActivationFunctions tanh = std::make_tuple(Internal::tanhFunction, Internal::tanhDerivate);
			static Internal::ActivationFunctions ReLU = std::make_tuple(Internal::ReLUFunction, Internal::ReLUDerivate);
			static Internal::ActivationFunctions LeakyReLU(const float& alpha)
			{
				return std::make_tuple(
					[&alpha](auto&& inputs) { return Internal::leakyReLUFunction(std::forward<decltype(inputs)>(inputs), alpha); },
					[&alpha](auto&& inputs) { return Internal::leakyReLUDerivate(std::forward<decltype(inputs)>(inputs), alpha); }
				);
			}

			/*static Internal::ActivationFunctions LeakyReLU(float& alpha)
			{
				return std::make_tuple(
					std::bind(leakyReLUFunction, std::placeholders::_1, alpha),
					std::bind(leakyReLUDerivate, std::placeholders::_1, alpha)
				);
			}*/
		}
}
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
			}
		
			static std::vector<float> identityFunction(const std::vector<float> & inputs)
			{
				return inputs;
			}

			static std::vector<float> identityDerivate(const std::vector<float> & inputs)
			{
				return std::vector<float>(inputs.size(), 1);
			}

			static std::vector<float> sigmoidFunction(const std::vector<float> & inputs)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), [](const float& z)->float {return 1.0f / (1.0f + std::exp(-z)); });
				return res;
			}

			static std::vector<float> sigmoidDerivate(const std::vector<float> & inputs)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), [](const float& z)->float {return (std::exp(-z) / powf((std::exp(-z) + 1.0f), 2)); });
				return res;
			}

			static std::vector<float> softmaxFunction(const std::vector<float> & inputs)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), [](const float& z)->float {return std::exp(z); });
				auto sum = std::accumulate(res.begin(), res.end(), 0.f);
				std::transform(res.begin(), res.end(), res.begin(), [&sum](const float& z)->float {return z / sum; });
				return res;
			}

			static std::vector<float> softmaxDerivate(const std::vector<float> & inputs)
			{
				std::vector<float> res(inputs.size());
				std::transform(inputs.begin(), inputs.end(), res.begin(), [](const float& z)->float {return std::exp(z); });
				auto sum = std::accumulate(res.begin(), res.end(), 0.f);
				std::transform(res.begin(), res.end(), res.begin(), [&sum](const float& z)->float {return z * (sum - 1) / (sum * sum); });
				return res;
			}

			static Internal::ActivationFunctions sigmoid = std::make_tuple(sigmoidFunction, sigmoidDerivate);
			static Internal::ActivationFunctions softmax = std::make_tuple(softmaxFunction, softmaxDerivate);
			static Internal::ActivationFunctions identity = std::make_tuple(identityFunction, identityDerivate);
		}
}
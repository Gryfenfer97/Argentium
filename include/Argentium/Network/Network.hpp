#pragma once
#include <Argentium/Network/BaseLayer.hpp>
#include <Argentium/Dataset.hpp>
#include <algorithm>

namespace Ag
{
	typedef std::vector<std::shared_ptr<Ag::LayerBuilder>> Topology;

	class Network
	{
	private:
		std::vector<std::unique_ptr<BaseLayer>> layers;


		std::vector<float> feedForward(const std::vector<float>& inputs);
		void backProp(const std::vector<float>& expectedOutput, const std::vector<float>& output);

		[[nodiscard]] std::vector<float> calculateCost(const std::vector<float>& expectedOutput, const std::vector<float>& output) const;
	public:
		Network() = default;

		explicit Network(const Topology& topology);

		[[nodiscard]] std::vector<float> evaluate(const std::vector<float>& inputs);
		float train(const DataSet set, std::size_t epochSize);
		float test(const DataSet& set);
	};
}
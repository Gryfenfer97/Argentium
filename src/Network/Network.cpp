#include <Argentium/Network/Network.hpp>
#include <iostream>

using namespace Ag;

std::vector<float> Network::feedForward(const std::vector<float>& inputs)
{
	std::vector<float> out = this->layers.front()->feedForward(inputs);

	for (auto it = this->layers.begin() + 1; it != this->layers.end(); ++it) {
		out = (*it)->feedForward(out);
	}
	return out;
}

void Network::backProp(const std::vector<float>& expectedOutput, const std::vector<float>& output)
{
	std::vector<float> cost = calculateCost(expectedOutput, output);
	for (auto it = std::rbegin(this->layers); it != std::rend(this->layers); ++it) {
		cost = (*it)->backProp(cost);
	}
}

std::vector<float> Network::calculateCost(const std::vector<float>& expectedOutput, const std::vector<float>& output) const
{
	std::vector<float> costs(output.size());
	std::transform(
		output.begin(),
		output.end(),
		expectedOutput.begin(),
		costs.begin(),
		[](const float& a, const float& b)->float {return -2 * (a - b); }
	);
	return costs;
}


Network::Network(const Topology& topology)
{
	this->layers.resize(topology.size()-1);
	for (std::size_t i = 1; i < topology.size(); i++)
	{
		this->layers[i-1] = std::move(topology[i]->createLayer(topology[i-1]->numberOfOutput));
	}
}

std::vector<float> Network::evaluate(const std::vector<float>& inputs)
{
	return feedForward(inputs);
}

float Network::train(DataSet set, const std::size_t epochSize)
{
	for(std::size_t i=0; i < epochSize; i++)
	{
		set.shuffle();
		for(std::size_t index = 0; index < set.size(); index++)
		{
			const auto& [input, output] = set[index];
			std::vector<float> result = feedForward(input);
			this->backProp(output, result);
		}
	}
	return 0.0f;
}

float Network::test(const DataSet& dataSet)
{
	float successRate = 0.f;
	for (std::size_t index = 0; index < dataSet.size(); index++) {
		const auto [input, output] = dataSet[index];
		std::vector<float> result = feedForward(input);
		
		if (std::max_element(std::begin(result), std::end(result)) - std::begin(result) == std::max_element(std::begin(output), std::end(output)) - std::begin(output)) {
			successRate++;
		}
		
	}
	return successRate / static_cast<float>(dataSet.size());
}

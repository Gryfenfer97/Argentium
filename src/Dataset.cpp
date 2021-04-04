#include <Argentium/Dataset.hpp>

using namespace Ag;

DataSet::DataSet(const std::size_t size) : set(size)
{
}

void DataSet::shuffle()
{
	std::random_device rd;
	std::mt19937 g{ rd() };
	std::shuffle(set.begin(), set.end(), g);
}

void DataSet::push(const std::vector<float>& input, const std::vector<float>& output)
{
	this->set.push_back(std::make_pair(input, output));
}

void DataSet::clear()
{
	this->set.clear();
}

std::size_t DataSet::size() const
{
	return this->set.size();
}

const std::pair<std::vector<float>, std::vector<float>>& DataSet::operator[](const std::size_t index)
{
	return this->set[index];
}

std::pair<std::vector<float>, std::vector<float>> DataSet::operator[](const std::size_t index) const
{
	return this->set[index];
}

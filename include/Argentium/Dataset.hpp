#pragma once
#include <vector>
#include <random>

namespace Ag
{
	namespace Internal
	{
		typedef std::vector<std::pair<std::vector<float>, std::vector<float>>> Set;
	}

	
	class DataSet
	{
	public:
		DataSet() = default;
		explicit DataSet(const std::size_t size);

		void shuffle();
		void push(const std::vector<float>& input, const std::vector<float>& output);
		void clear();
		[[nodiscard]] std::size_t size() const;

		const std::pair<std::vector<float>, std::vector<float>>& operator[](const std::size_t index);
		std::pair<std::vector<float>, std::vector<float>> operator[](const std::size_t index) const;

	private:
		Internal::Set set;
	};
	
}


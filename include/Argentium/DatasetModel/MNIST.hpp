#pragma once
#include <Argentium/Dataset.hpp>
#include <fstream>
#include <cassert>

namespace Ag
{
	namespace DataSetModel {
		class MNIST : public Ag::DataSet
		{
		private:
			template<typename T>
			T read_n_bytes(std::ifstream& file);

			template<typename T>
			static T swap_endiant(T u);

		public:
			MNIST() = default;
			MNIST(const std::string& labelFileName, const std::string& imageFileName, std::size_t classesNumber = 10);
		};
	}
}

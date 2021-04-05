#include <Argentium/DatasetModel/MNIST.hpp>
#include <climits>

using namespace Ag::DataSetModel;

MNIST::MNIST(const std::string& labelFileName, const std::string& imageFileName, std::size_t classesNumber)
{
	/* Image data set */
	std::ifstream file{ imageFileName, std::ios::out | std::ios::binary };
	// assert(file && "Cannot read MNIST Image file");
	if (!file.is_open()) {
		throw std::runtime_error("Cannot open the MNIST image file");
	}

	[[maybe_unused]] auto magicNumber = read_n_bytes<uint32_t>(file);
	assert(magicNumber == 2051 && "your image file is not a MNIST file");


	auto imagesNumber = read_n_bytes<uint32_t>(file);
	auto rowsNumber = read_n_bytes<uint32_t>(file);
	auto columnsNumber = read_n_bytes<uint32_t>(file);



	/* Label data set */
	std::ifstream labelFile{ labelFileName, std::ios::out | std::ios::binary };
	if (!labelFile.is_open()) {
		throw std::runtime_error("Cannot open the MNIST label file");
	}

	[[maybe_unused]] auto labelMagicNumber = read_n_bytes<uint32_t>(labelFile);
	assert(labelMagicNumber == 2049 && "your label file is not a MNIST file");

	[[maybe_unused]] auto labelsNumber = read_n_bytes<uint32_t>(labelFile);
	assert(labelsNumber == imagesNumber && "The number of images and labels does not correspond");




	/* Model creation */
	for (unsigned image_id = 0; image_id < imagesNumber; image_id++) {

		//read the label
		auto label = read_n_bytes<uint8_t>(labelFile);

		std::vector<float> result(classesNumber);
		result[static_cast<int>(label)] = 1;
		std::vector<float> image;

		for (unsigned i = 0; i < rowsNumber * columnsNumber; i++) {
			auto pixel = read_n_bytes<uint8_t>(file);
			pixel = (pixel != 0x00) ? 0xff : pixel;
			image.push_back(static_cast<float>(pixel) / 255.f);
		}
		push(image, result);

	}

	file.close();
	labelFile.close();


}


template<typename T>
inline T MNIST::read_n_bytes(std::ifstream& file)
{
	T number;
	file.read((char*)&number, sizeof(T));
	number = MNIST::swap_endiant<T>(number);
	return number;
}

template<typename T>
T MNIST::swap_endiant(T u){
	static_assert (CHAR_BIT == 8, "CHAR_BIT != 8");

	union // put a union to access to u by unsigned char
	{
		T u;
		unsigned char u8[sizeof(T)];
	} source, dest;

	source.u = u;

	for (size_t k = 0; k < sizeof(T); k++)
		dest.u8[k] = source.u8[sizeof(T) - k - 1];

	return dest.u;
}




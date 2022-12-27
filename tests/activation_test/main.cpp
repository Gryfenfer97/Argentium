#include <gtest/gtest.h>
#include <Argentium/Network/Activation.hpp>

TEST(Activation, softmax)
{
	const std::vector<float> inputs = {1, 3, 2.5, 5, 4, 2};
	std::vector<float> res = Ag::Activation::Internal::softmaxFunction(inputs);
	std::transform(
		res.begin(), res.end(), res.begin(), [](const float &z) -> auto{ return static_cast<float>(std::truncf(z * 100)) / 100; }); // trunc at 10^-2
	const std::vector<float> expectedResult = {0.01f, 0.08f, 0.04f, 0.60f, 0.22f, 0.03f};
	EXPECT_EQ(res, expectedResult);
}

TEST(Activation, identity)
{
	const std::vector<float> inputs = {1, 3, 2.5, 5, 4, 2};
	std::vector<float> res = Ag::Activation::Internal::identityFunction(inputs);
	EXPECT_EQ(res, inputs);
}

// TODO : Test for sigmoid function

int main(int argc, char *argv[])
{

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
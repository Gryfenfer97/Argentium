#include <gtest/gtest.h>
#include <Argentium/Network/Network.hpp>
#include <Argentium/Dataset.hpp>
#include <Argentium/DatasetModel/MNIST.hpp>
#include <chrono>
#include <Argentium/Network/Input.hpp>
#include <Argentium/Network/FullyConnected.hpp>
#include <Argentium/Network/Activation.hpp>

TEST(Dataset, xor)
{
    Ag::DataSet data;
    data.push({ 0., 0. }, { 0. });
    data.push({ 0., 1. }, { 1. });
    data.push({ 1., 0. }, { 1. });
    data.push({ 1., 1. }, { 0. });
    const std::vector<unsigned> topology = { 2,4,4,1 };

	
    std::vector<std::shared_ptr<Ag::LayerFactory>> t = {
    	Ag::FullyConnectedLayer(2,Ag::Activation::sigmoid),
    	Ag::FullyConnectedLayer(4,Ag::Activation::sigmoid),
        Ag::FullyConnectedLayer(4,Ag::Activation::sigmoid),
        Ag::FullyConnectedLayer(1,Ag::Activation::sigmoid)

    };

    Ag::Network net(t);

	
    net.train(data, 150000);

	
    const auto successRate = net.test(data);
    EXPECT_EQ(successRate, 1);
    //EXPECT_EQ(1, 1);
}

TEST(Dataset, mnist)
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    Ag::DataSetModel::MNIST trainData(std::string("./datasets/MNIST/train-labels.idx1-ubyte"), std::string("./datasets/MNIST/train-images.idx3-ubyte"));
    std::vector<std::shared_ptr<Ag::LayerFactory>> topology = {
        Ag::InputLayer(784),
        Ag::FullyConnectedLayer(230,Ag::Activation::tanh),
        Ag::FullyConnectedLayer(10,Ag::Activation::sigmoid)
    };

    Ag::Network net{ topology };

    net.train(trainData, 1);


    Ag::DataSetModel::MNIST testData(std::string("./datasets/MNIST/t10k-labels.idx1-ubyte"), std::string("./datasets/MNIST/t10k-images.idx3-ubyte"));
    
    const auto successRate = net.test(testData);
	end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time)
        << "elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "successRate : " << successRate << std::endl;
    EXPECT_GT(successRate, 0.90f);
}


TEST(Dataset, fashionMnist)
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
	Ag::DataSetModel::MNIST trainData(std::string("./datasets/Fashion-MNIST/train-labels-idx1-ubyte"), std::string("./datasets/Fashion-MNIST/train-images-idx3-ubyte"));
    const std::vector<std::shared_ptr<Ag::LayerFactory>> topology = {
        Ag::InputLayer(784),
        Ag::FullyConnectedLayer(230,Ag::Activation::sigmoid),
        Ag::FullyConnectedLayer(10,Ag::Activation::softmax)
    };

    Ag::Network net{ topology };

    net.train(trainData, 1);


    Ag::DataSetModel::MNIST testData(std::string("./datasets/Fashion-MNIST/t10k-labels-idx1-ubyte"), std::string("./datasets/Fashion-MNIST/t10k-images-idx3-ubyte"));

    const auto successRate = net.test(testData);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time)
        << "elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "successRate : " << successRate << std::endl;
    EXPECT_GT(successRate, 0.80f);
}

/*TEST(Dataset, emnist)
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    std::cout << "hello\n";
    Ag::DataSetModel::MNIST trainData(std::string("datasets/EMNIST/train/emnist-letters-train-labels-idx1-ubyte"), std::string("datasets/EMNIST/train/emnist-letters-train-images-idx3-ubyte"), 27);
    const std::vector<std::shared_ptr<Ag::LayerFactory>> topology = {
        Ag::InputLayer(784),
        Ag::FullyConnectedLayer(230,Ag::Activation::sigmoid),
        Ag::FullyConnectedLayer(27,Ag::Activation::sigmoid)
    };

    Ag::Network net{ topology };

    net.train(trainData, 1);


    Ag::DataSetModel::MNIST testData(std::string("datasets/EMNIST/test/emnist-letters-test-labels-idx1-ubyte"), std::string("datasets/EMNIST/test/emnist-letters-test-images-idx3-ubyte"), 27);

    const auto successRate = net.test(testData);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time)
        << "elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "successRate : " << successRate << std::endl;
    EXPECT_GT(successRate, 0.70f);
}*/


int main(int argc, char* argv[]){

	testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
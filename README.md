# Argentium
![](https://github.com/Gryfenfer97/Argentium/workflows/dataset_test/badge.svg?barnch=master)

A remake of Bromure, A deep learning laboratory

Argentium is a deep learning library in c++17. The goal is to have a library powerful enough for a simple use like in a OCR projects, but also scalable to allow everyone to test a new feature in a neural network without recoding everything.

## How to use

Here is an example for MNIST dataset

```cpp
#include <iostream>
#include <Argentium/Network/Network.hpp>
#include <Argentium/DatasetModel/MNIST.hpp>

int main(){
  Ag::DataSetModel::MNIST trainData(
    std::string("./datasets/MNIST/train-labels.idx1-ubyte"),
    std::string("./datasets/MNIST/train-images.idx3-ubyte")
  ); //load the training data
  
  std::vector<std::shared_ptr<Ag::LayerFactory>> topology = {
        Ag::InputLayer(784),
        Ag::FullyConnectedLayer(230,Ag::Activation::tanh),
        Ag::FullyConnectedLayer(10,Ag::Activation::sigmoid)
  }; // Create the topology for your network
  
  Ag::Network net{ topology }; // Generate the network
  net.train(trainData, 1); //training with epoch = 1

  Ag::DataSetModel::MNIST testData(
    std::string("./datasets/MNIST/t10k-labels.idx1-ubyte"),
    std::string("./datasets/MNIST/t10k-images.idx3-ubyte")
  ); // load the testing data
  const auto accuracy = net.test(testData); // test
  
  std::cout << "Accuracy : " << accuracy << std::endl;
}
```

## Building

### Linux

```bash
# clone the project
~$ git clone https://github.com/Gryfenfer97/Argentium.git
~$ cd Argentium

~$ cd build
~$ cmake -G"Unix Makefiles" ./.. -DCMAKE_BUILD_TYPE=Release
~$ cmake --build . --config Release
```

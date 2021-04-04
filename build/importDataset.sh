#!/bin/bash

# MNIST
train_images_url="yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
train_labels_url="yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
test_images_url="yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
test_labels_url="yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"


mkdir "./tests/dataset_tests/datasets/mnist/{train, test}"

curl -o ./tests/dataset_tests/datasets/mnist/train/train-images-idx3-ubyte.gz $train_images_url
gzip -d ./tests/dataset_tests/datasets/mnist/train/train-images-idx3-ubyte.gz
curl -o ./tests/dataset_tests/datasets/mnist/train/train-labels-idx1-ubyte.gz $train_labels_url
gzip -d ./tests/dataset_tests/datasets/mnist/train/train-labels-idx1-ubyte.gz

curl -o ./tests/dataset_tests/datasets/mnist/test/t10k-images-idx3-ubyte.gz $test_images_url
gzip -d ./tests/dataset_tests/datasets/mnist/test/t10k-images-idx3-ubyte.gz
curl -o ./tests/dataset_tests/datasets/mnist/test/t10k-labels-idx1-ubyte.gz $test_labels_url
gzip -d ./tests/dataset_tests/datasets/mnist/test/t10k-labels-idx1-ubyte.gz
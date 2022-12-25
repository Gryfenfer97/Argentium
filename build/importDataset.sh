#!/bin/bash
rm -r tests/dataset_tests/datasets/
git clone --no-tags --depth 1 "https://github.com/MatthieuHernandez/Datasets-for-Machine-Learning.git" ./tests/bin/datasets
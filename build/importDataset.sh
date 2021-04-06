#!/bin/bash
rm -rf "tests/dataset_tests/datasets"
git clone --no-tags --depth 1 "https://github.com/MatthieuHernandez/Datasets-for-Machine-Learning.git" ./tests/dataset_tests/datasets
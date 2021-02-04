# Row-stochastic Tensor DEDICOM

This directory provides the source code to the paper
"**[Interpretable Topic Extraction and Word Embedding Learning Using Non-Negative Tensor DEDICOM](https://www.mdpi.com/2504-4990/3/1/7)**" [[1]](#1).

## Installation

1. Clone this repository.
1. Create an empty python environment, e.g. using conda: ``` conda create -n text_tensor_dedicom python=3.7```.
1. Pip install the *text_tensor_dedicom* package by navigating into the cloned repo (directory of the *setup.py* file) and execute ```pip install .```.

## Run Training Pipeline
1. Edit one of the *config.yaml* files located in the `configs` directory. Each config file represents a dataset, which are located at `data`. Select an output directory and a training setup.
1. Run the main script for the Amazon reviews dataset via ``` python run.py  --config configs/config_amazon_reviews.yaml```. Optionally, select the number of processes to utilize multiprocessing (e.g. ``` python run.py --num-processes 4```) and train different setups simultaneously.


## References
<a id="1">[1]</a>
Hillebrand, Biesner et. al. (2021).
Interpretable Topic Extraction and Word Embedding Learning Using Non-Negative Tensor DEDICOM.
Published at [Machine Learning and Knowledge Extraction](https://www.mdpi.com/journal/make) Journal.

# Row-stochastic DEDICOM

This repository provides the source code to the paper "**Interpretable Topic Extraction and Word Embedding Learning using row-stochastic DEDICOM**" [[1]](#1).

## Installation

1. Clone this repository.
1. Create an empty python environment, e.g. using conda: ``` conda create -n text_dedicom python=3.7```.
1. Pip install the *text_dedicom* package by navigating into the cloned repo (directory of the *setup.py* file) and execute ```pip install .```.

## Run Training Pipeline
1. Edit the *config.yaml* file. Select an output directory and a training setup.
1. Run the main script via ``` python run.py```. Optionally, select the number of processes to utilize multiprocessing (e.g. ``` python run.py --num-processes 4```) and train different setups simultaneously.


## References
<a id="1">[1]</a> 
Hillebrand, Biesner et. al. (2020). 
Interpretable Topic Extraction and Word Embedding Learning using row-stochastic DEDICOM. 
Accepted at CD-MAKE.

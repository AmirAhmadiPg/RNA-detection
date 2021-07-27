# Welcome to the loyal_customers wiki!
## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Hyperparameters](#Hyperparameters)

## this was a project for ict6 challenge
# General info
in this project, we are trying to solve the problem of classification coding and non-coding with Recurrent neural networks and Multi-head Attention models

# Hyperparameters

**Positional embeding:**
PE(pos, 2i) = sin(pos / 10000 ** 2i / d model)
PE(pos, 2i+1) = cos(pos / 10000 ** 2i / d model)

**Embedding inputs:**
input dims = 5
output dims = 6

**Attention layers:**
1 Head attention

**RNN layers:**
GRU with 2 cells

**Hidden layers:**
1 Dense layer befor RNN with units: (10, 15)
2 Dense layer after RNN with units: 2

**classification layer:**
Dense with 1 unit


**Compile Model:**
Loss: Binary crossentropy
Optimizer: Adam

**Optimizer Paramiters:**
learning rate: 0.009
beta 1: 0.6
beta 2: 0.6
opsilon: 1e-07

## Technologies
project is created with:

* language: **python**
* python version: **3.8.8**
* libs: **Tensorflow, Keras, Pytorch, Bio, pandas, Numpy, Matplotlib, Tensorboard**

## Setup
To run this project, install it locally using python:
```
$ cd /addres/to/project/folder
$ python3 RNA_Detection_AttentionBase_keras_V.py #For tensorflow version
$ python3 data.py #For pytorch version 
```
**Warning Pytorch version is without attention layers**

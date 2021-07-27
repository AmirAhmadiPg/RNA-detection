# Welcome to the loyal_customers wiki!
## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [output example](#output-example)

## this was a project for ict6 challenge
about this challenge:
The ICT challenge is **the most popular** tournament between Programmers, with questions in different parts of computer science like: **A.i, Blockchain, web design, back-end development, and more...**
**We got 3rd 🥉 place in this tournament**

Country: **Iran**
Date: **15th july 2021**
Organizer: **ICT faculty, Sharif university of technology,  Tehran, Iran**
Organize Website: http://ictchallenge.sharif.ir/

# General info
this project will **cluster** 4 classes of customers: **Legendary/Epic/Rare/Common customer**

These clusters are trained on the **ict6 challenge dataset** and you can download it from the repository (trade.csv).

we choose to use **karmozd(or tx)** and the **number of trades** for features to train the model

### normilization:

**upper limit = Q3 + 1.5 * IQR**
**lower limit = Q3 - 1.5 * IQR**

**every data upper or lower than this threshold value have to remove from dataset**
**lowerlimit < data < upperlimit**


### hyperparameters for clustering:

**number of clusters = 4**


## Technologies
project is created with:

* language: **python**
* python version: **3.8.8**
* libs: **sklearn, matplotlib and pandas**

## Setup
To run this project, install it locally using python:
```
$ cd /addres/to/project/folder
$ python3 Main.py
```

## output example
### output of metric 1
![alt text](./output_metric_1.png?raw=true)

### output of metric 2
![alt text](./output_metric_2.png?raw=true)

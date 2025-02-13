# GraphFVD: Property Graph-based Fine-grained Vulnerability Detection
This is an official implementation of our paper "[GraphFVD: Property Graph-based Fine-grained Vulnerability Detection](https://www.sciencedirect.com/science/article/abs/pii/S0167404825000392)".
# Overview 
In this repository, you will find a Python implementation of our GraphFVD. As described in our paper, GraphFVD is a novel property graph-based fine-grained vulnerability detection approach. Our approach extracts property graph-based slices from the Code Property Graph and introduces a Hierarchical Attention Graph Convolutional Network to learn graph embeddings. 
# Requirements
- Python 3.7.2
- Pytorch 1.9.1
- Transformer 4.30.2
- Tokenizers 0.13.3
# Training and Evaluation
```
python my_preprocess.py
```
This command preprocesses the dataset and splits it into training, validation, and testing sets. 
```
python run.py
```
This command is used to train and evaluate GraphFVD model.


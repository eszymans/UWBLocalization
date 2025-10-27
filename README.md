# UWB Localization Correction Using Neural Networks

### Project Objective
The goal of this project is to develop a program that improves inaccurate robot localization measurements using Ultra-Wideband (UWB) technology. The correction is performed using an artificial neural network. Various network architectures and training methods were tested to identify the most effective configuration.

### Technologies Used
Programming Language: Python
Machine Learning Library: PyTorch
Data Scaling Method: Min-Max Normalization

### Network Variants Tested
| Activation Function | Hidden Layer Neurons | Weight Initialization | Weight Range         | Optimizer | Learning Rate |
|----------------------|----------------------|------------------------|--------------------|------------|----------------|
| ReLU                 | 64                   | xavier_uniform         | [-0.3015, 0.3015]  | Adam       | 0.01           |
| Sigmoid              | 16                   | xavier_uniform         | [-0.577, 0.577]    | Adam       | 0.01           |
| Tanh                 | 64                   | xavier_uniform         | [-0.3015, 0.3015]  | Adam       | 0.01           |


### Project Structure
Code
├── data_preparation.py   # Data loading and preprocessing
├── MLP.py                # Neural network architecture and training logic
├── main.py               # Main script to run the training and evaluation
└── README.md             # Project documentatio

### Authors
Alicja Bartczak
Edyta Szymańska

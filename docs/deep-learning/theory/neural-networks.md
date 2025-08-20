---
sidebar_position: 1
---

# Neural Network Basics

## Introduction

Neural networks are computational models inspired by biological neural networks in the human brain. They form the foundation of modern deep learning and are capable of learning complex patterns from data.

## Key Concepts

### 1. Artificial Neuron (Perceptron)

The basic building block of a neural network is the artificial neuron, also called a perceptron.

**Mathematical Representation:**
```
y = f(Σ(w_i * x_i) + b)
```

Where:
- `x_i` are the input features
- `w_i` are the weights
- `b` is the bias term
- `f()` is the activation function
- `y` is the output

### 2. Layers

Neural networks are organized in layers:

- **Input Layer**: Receives the raw input data
- **Hidden Layers**: Process the data through weighted connections
- **Output Layer**: Produces the final prediction

### 3. Feedforward Propagation

The process of computing outputs from inputs:

1. Input data flows through the network
2. Each layer applies weights, biases, and activation functions
3. Final output is produced at the output layer

## Types of Neural Networks

### 1. Feedforward Neural Networks (FNN)
- Simplest type
- Information flows in one direction
- No cycles or loops

### 2. Convolutional Neural Networks (CNN)
- Specialized for grid-like data (images)
- Uses convolutional layers
- Excellent for computer vision tasks

### 3. Recurrent Neural Networks (RNN)
- Designed for sequential data
- Has memory of previous inputs
- Used in NLP, time series analysis

### 4. Long Short-Term Memory (LSTM)
- Advanced RNN architecture
- Better at capturing long-term dependencies
- Addresses vanishing gradient problem

## Training Process

### 1. Forward Pass
- Compute predictions using current weights
- Calculate loss/error

### 2. Backward Pass (Backpropagation)
- Compute gradients of loss with respect to weights
- Update weights using optimization algorithm

### 3. Iteration
- Repeat until convergence or stopping criteria

## Common Architectures

### Multi-Layer Perceptron (MLP)
```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer
```

### Deep Neural Network
- Multiple hidden layers
- Can learn hierarchical representations
- Requires more data and computational resources

## Key Considerations

### 1. Depth vs Width
- **Depth**: Number of layers
- **Width**: Number of neurons per layer
- Trade-off between expressiveness and training difficulty

### 2. Overfitting
- Model performs well on training data but poorly on new data
- Solutions: regularization, dropout, early stopping

### 3. Vanishing/Exploding Gradients
- Common problem in deep networks
- Solutions: proper weight initialization, batch normalization

## Implementation Example

```python
import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

## Practice Questions

1. **What is the difference between a perceptron and a multi-layer perceptron?**
2. **Explain the concept of backpropagation in your own words.**
3. **Why do we need activation functions in neural networks?**
4. **What happens if you remove the bias term from a neural network?**
5. **How would you determine the optimal number of layers for a given problem?**

## Further Reading

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [Deep Learning Book Chapter 6](https://www.deeplearningbook.org/contents/mlp.html)
- [3Blue1Brown Neural Network Series](https://www.youtube.com/watch?v=aircAruvnKk)

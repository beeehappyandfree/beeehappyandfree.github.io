---
sidebar_position: 2
---

# Activation Functions

## Introduction

Activation functions are crucial components of neural networks that introduce non-linearity, enabling networks to learn complex patterns. Without activation functions, neural networks would only be able to learn linear relationships.

## Why Activation Functions?

### Linear vs Non-linear
- **Linear functions**: Can only model linear relationships
- **Non-linear functions**: Enable modeling of complex, non-linear patterns
- **Universal approximation**: Non-linear activation functions allow neural networks to approximate any continuous function

## Common Activation Functions

### 1. Sigmoid (Logistic)

**Formula:**
```
σ(x) = 1 / (1 + e^(-x))
```

**Properties:**
- Output range: (0, 1)
- Smooth and differentiable
- Historically popular for binary classification

**Problems:**
- Vanishing gradient problem
- Not zero-centered
- Saturation for large inputs

**Use Cases:**
- Binary classification output layer
- Historical significance (not recommended for hidden layers)

### 2. Tanh (Hyperbolic Tangent)

**Formula:**
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Properties:**
- Output range: (-1, 1)
- Zero-centered
- Smooth and differentiable

**Advantages over Sigmoid:**
- Better gradient flow
- Zero-centered output

**Problems:**
- Still suffers from vanishing gradient
- Saturation for large inputs

### 3. ReLU (Rectified Linear Unit)

**Formula:**
```
ReLU(x) = max(0, x)
```

**Properties:**
- Output range: [0, ∞)
- Simple and computationally efficient
- No saturation for positive inputs

**Advantages:**
- Mitigates vanishing gradient problem
- Fast computation
- Sparse activation (many neurons output 0)

**Problems:**
- Dying ReLU problem (neurons can become permanently inactive)
- Not zero-centered

**Variants:**
- **Leaky ReLU**: `f(x) = max(αx, x)` where α is small (e.g., 0.01)
- **Parametric ReLU (PReLU)**: α is learned during training
- **ELU**: `f(x) = x if x > 0 else α(e^x - 1)`

### 4. Softmax

**Formula:**
```
softmax(x_i) = e^(x_i) / Σ(e^(x_j))
```

**Properties:**
- Outputs probability distribution (sums to 1)
- Used in multi-class classification
- Maintains relative order of inputs

**Use Cases:**
- Multi-class classification output layer
- Attention mechanisms

### 5. Swish/SiLU

**Formula:**
```
swish(x) = x * σ(x)
```

**Properties:**
- Smooth and differentiable
- Self-gated (output depends on input)
- Often outperforms ReLU

**Advantages:**
- No dying neuron problem
- Smooth gradients
- Better performance in deep networks

## Choosing Activation Functions

### Hidden Layers
- **ReLU**: Default choice for most cases
- **Leaky ReLU**: If concerned about dying ReLU
- **Swish**: For better performance (computational cost)
- **Tanh**: For bounded outputs

### Output Layers
- **Sigmoid**: Binary classification
- **Softmax**: Multi-class classification
- **Linear**: Regression
- **ReLU**: Positive regression

## Implementation Examples

### PyTorch
```python
import torch
import torch.nn as nn

# Different activation functions
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(0.01)
tanh = nn.Tanh()
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)
swish = nn.SiLU()  # or nn.Swish()
```

### Custom Activation Function
```python
class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()
    
    def forward(self, x):
        return torch.where(x > 0, x, 0.1 * x)  # Leaky ReLU
```

## Mathematical Properties

### Derivatives

**Sigmoid:**
```
dσ/dx = σ(x) * (1 - σ(x))
```

**Tanh:**
```
d(tanh)/dx = 1 - tanh²(x)
```

**ReLU:**
```
d(ReLU)/dx = 1 if x > 0 else 0
```

**Swish:**
```
d(swish)/dx = σ(x) + x * σ(x) * (1 - σ(x))
```

## Best Practices

1. **Start with ReLU**: Default choice for hidden layers
2. **Use Leaky ReLU**: If you encounter dying ReLU problem
3. **Try Swish**: For potentially better performance
4. **Avoid Sigmoid/Tanh**: In hidden layers of deep networks
5. **Choose output activation**: Based on problem type

## Practice Questions

1. **Why can't we use linear activation functions in hidden layers?**
2. **What is the dying ReLU problem and how can it be solved?**
3. **Compare the computational efficiency of ReLU vs Sigmoid.**
4. **When would you choose Tanh over ReLU?**
5. **Explain why Softmax is used in multi-class classification.**
6. **What are the advantages of Swish over ReLU?**

## Further Reading

- [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
- [Swish: A Self-Gated Activation Function](https://arxiv.org/abs/1710.05941)
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) (ReLU paper)

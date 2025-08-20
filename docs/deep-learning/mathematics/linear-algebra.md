---
sidebar_position: 1
---

# Linear Algebra Foundations

## Introduction

Linear algebra is the mathematical foundation of deep learning. Understanding vectors, matrices, and their operations is essential for comprehending how neural networks process data and learn.

## Key Concepts

### 1. Vectors

**Definition:** Ordered lists of numbers representing points in space.

**Notation:**
```
v = [v₁, v₂, ..., vₙ]
```

**Properties:**
- **Magnitude**: ||v|| = √(v₁² + v₂² + ... + vₙ²)
- **Direction**: Unit vector = v / ||v||
- **Dot Product**: v · w = Σ(vᵢ * wᵢ)

### 2. Matrices

**Definition:** Rectangular arrays of numbers organized in rows and columns.

**Notation:**
```
A = [a₁₁  a₁₂  a₁₃]
    [a₂₁  a₂₂  a₂₃]
    [a₃₁  a₃₂  a₃₃]
```

**Dimensions:** m × n matrix has m rows and n columns

### 3. Matrix Operations

#### Addition/Subtraction
```
C = A ± B where cᵢⱼ = aᵢⱼ ± bᵢⱼ
```

#### Scalar Multiplication
```
C = αA where cᵢⱼ = α * aᵢⱼ
```

#### Matrix Multiplication
```
C = AB where cᵢⱼ = Σ(aᵢₖ * bₖⱼ)
```

**Important:** Matrix multiplication is not commutative (AB ≠ BA)

### 4. Special Matrices

#### Identity Matrix (I)
```
I = [1  0  0]
    [0  1  0]
    [0  0  1]
```

#### Transpose (A^T)
```
A^T = [a₁₁  a₂₁  a₃₁]
      [a₁₂  a₂₂  a₃₂]
      [a₁₃  a₂₃  a₃₃]
```

#### Inverse (A^(-1))
```
AA^(-1) = A^(-1)A = I
```

## Deep Learning Applications

### 1. Linear Transformations

**Neural Network Layer:**
```
y = Wx + b
```

Where:
- W is the weight matrix
- x is the input vector
- b is the bias vector
- y is the output vector

### 2. Batch Processing

**Multiple samples:**
```
Y = XW^T + b
```

Where:
- X is the input matrix (batch_size × features)
- W is the weight matrix (output_features × input_features)
- Y is the output matrix (batch_size × output_features)

### 3. Convolution as Matrix Multiplication

**1D Convolution:**
```
y = conv(x, w) = X * w
```

Where X is a Toeplitz matrix constructed from x.

## Eigenvalues and Eigenvectors

### Definition
For matrix A, if Av = λv, then:
- λ is an eigenvalue
- v is an eigenvector

### Applications in Deep Learning
- **Principal Component Analysis (PCA)**
- **Singular Value Decomposition (SVD)**
- **Understanding model dynamics**

## Singular Value Decomposition (SVD)

### Formula
```
A = UΣV^T
```

Where:
- U: Left singular vectors (orthogonal)
- Σ: Singular values (diagonal matrix)
- V: Right singular vectors (orthogonal)

### Applications
- **Dimensionality reduction**
- **Matrix approximation**
- **Understanding model complexity**

## Vector Spaces and Subspaces

### Vector Space Properties
1. **Closure under addition**
2. **Closure under scalar multiplication**
3. **Existence of zero vector**
4. **Existence of additive inverse**

### Subspaces
- **Span**: Set of all linear combinations
- **Basis**: Linearly independent spanning set
- **Dimension**: Number of basis vectors

## Norms and Metrics

### Vector Norms

**L1 Norm (Manhattan):**
```
||x||₁ = Σ|xᵢ|
```

**L2 Norm (Euclidean):**
```
||x||₂ = √(Σxᵢ²)
```

**L∞ Norm (Maximum):**
```
||x||∞ = max|xᵢ|
```

### Matrix Norms

**Frobenius Norm:**
```
||A||_F = √(ΣΣ|aᵢⱼ|²)
```

## Implementation Examples

### NumPy
```python
import numpy as np

# Vector operations
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])
dot_product = np.dot(v, w)
norm = np.linalg.norm(v)

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.matmul(A, B)  # or A @ B

# Eigenvalues and eigenvectors
eigenvals, eigenvecs = np.linalg.eig(A)

# SVD
U, S, Vt = np.linalg.svd(A)
```

### PyTorch
```python
import torch

# Tensor operations
v = torch.tensor([1., 2., 3.])
w = torch.tensor([4., 5., 6.])
dot_product = torch.dot(v, w)
norm = torch.norm(v)

# Matrix operations
A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])
C = torch.matmul(A, B)  # or A @ B

# SVD
U, S, V = torch.svd(A)
```

## Computational Considerations

### 1. Memory Efficiency
- **Sparse matrices** for large, sparse data
- **Matrix factorization** for memory constraints

### 2. Numerical Stability
- **Condition number**: Measure of matrix stability
- **Regularization**: Adding small values to diagonal

### 3. Computational Complexity
- **Matrix multiplication**: O(n³) for n×n matrices
- **Strassen's algorithm**: O(n^2.807)
- **Coppersmith-Winograd**: O(n^2.376)

## Practice Questions

1. **What is the difference between a vector and a matrix?**
2. **Why is matrix multiplication not commutative?**
3. **Explain the concept of eigenvalues and eigenvectors.**
4. **How does SVD help in dimensionality reduction?**
5. **What is the relationship between L1 and L2 norms?**
6. **How would you implement a linear transformation in PyTorch?**

## Further Reading

- [Linear Algebra for Deep Learning](https://www.deeplearningbook.org/contents/linear_algebra.html)
- [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) by 3Blue1Brown
- [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

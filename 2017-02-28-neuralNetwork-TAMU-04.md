---
layout: post
title: Neural Network TAMU 04 Multi-Layer Perceptrons      
category: machine-learning

---


# Introduction  


Networks consist of input, hidden and output layers.  


Popular learning algorithm is the **error backpropagation algorithm**.  


- Forward pass: activate the network, layer by layer  
- Backward pass: error signal backpropagates from output to hidden and hidden to input.  


## Activation function  


sigmoid:  
\\[\phi(v)=\frac{1}{1+\exp(-v)}\\]  
\\[\frac{d\phi(v)}{dv} = \phi(v) (1-\phi(v))\\]  


other:  
\\[\tanh(v) = \frac{1 - \exp(-2v)}{1 + \exp(-2v)}\\]  



# Backpropagation  


## Error Gradient  


For $$n$$ input-output pairs $$\\{(x_k,d_k)\\}_{k=1}^{n}$$:  


\\[\frac{\partial E}{\partial w_i} = \frac{\partial }{\partial w_i} \left(\frac{1}{2} \sum_k (d_k - y_k)^2\right) \\\\ \hspace{0.6 cm} = -\sum_k (d_k-y_k) \frac{\partial y_k}{\partial v_k}\frac{\partial v_k}{\partial w_i} \\\\ =  - \sum_k (d_k-y_k) \centerdot y_k (1-y_k) \centerdot x_{i,k}\\]  

## Algorithm  

Initialize all weights to small random numbers.  

Until satisfied, do:  


1. output  unit $$j$$  
    - Error: $$\delta_j \leftarrow y_j (1-y_j)(d_j - y_j)$$  
2. hidden unit $$h$$  
    - Error: $$\delta_h \leftarrow y_h (1-y_h)\sum_{j \in outputs}w_{jh} \delta_j$$  
3. update each network weight $$w_{i,j}$$  
    - $$w_{ji} \leftarrow w_{ji} + \Delta w_{ji}$$
    - $$\Delta w_{ji} = \eta \delta_j x_i$$


Weight is defined as $$w_{j \leftarrow i}$$.  


## Derivation of $$\Delta w$$  


\\[\Delta w_{ji} = - \eta \frac{\partial E}{\partial w_{ji}} \\\\ = - \eta \frac{\partial E}{\partial v_{j}} \frac{\partial v_{j}}{\partial w_{ji}} \\\\ = \eta \left[ y_j (1-y_j) \sum_{k \in Downstream(j)} \delta_k w_{kj}\right]x_i\\]  
\\[\underbrace{\Delta w_{ji}(n)}_{\text{weight correction}} = \underbrace{\eta}_{\text{learning rate}} \centerdot \underbrace{\delta_j(n)}_{\text{local gradient}} \centerdot \underbrace{y_i(n)}_{\text{input signal}}\\]  


## Learning rate and momentum  


Introduce momentum to overcome problem caused by smaller and larger learning rate.  


\\[\Delta w_{ji}(n) = \eta \delta_j(n)y_i (n) + \alpha \Delta w_{ji}(n-1)\\]  
\\[\Delta w_{ji}(n) = - \eta \sum_{t=0}^{n} \alpha^{n-t} \frac{E(t)}{w_{ji}(t)}\\]  


- When successive $$\frac{E(t)}{w_{ji}(t)}$$ take the same sign: weight update is accelerated (speed up downhill)  
- When successive $$\frac{E(t)}{w_{ji}(t)}$$ take the same sign: weight update is damped (stabilize oscillation)  


# Sequential (online) vs. Batch  


- sequential 
    - help avoid local minima
    - hard to establish theoretical convergence  
- batch
    - accurate estimate of the gradient


# Overfitting  


- Training set error and validation set error  
- Early stopping ensures good performance on unobserved samples.  
- Solution: weight decay, use of validation sets, use of k-fold cross-validation  


# Recurrent Networks


- Sequence recognition  
- Store tree structure
- Can be trained with plain bp
- Represent a stack (push & pop)  


# Universal approximation theorem  


MLP can be seen as performing nonlinear input-output mapping.  


For a case with one hidden layer:  
\\[F(x_1,...,x_{m_0})=\sum_{i=1}^{m_1} \alpha_i \phi \left( \sum_{j=1}^{m_0} w_{ij}x_j +b_i\right)\\]  


To have:  
\\[\vert F(x_1,...,x_{m_0}) - F(x_1,...,x_{m_0})\vert \lt \epsilon\\]  


- Imply that one hidden layer is sufficient, but may not be optimal.  


# Generalization  


Smoothness in the mapping is desired, and this is related to criteria like Occam's razor.  


Affected by 3 factors:  


- Size of the training set, and how representative they are.  
- The architecture of the network
- Physical complexity of the problem  


Sample complexity and VC dimension are related.  


# Heuristic for Accelerating Convergence  


- Separate learning rate for each tunable weight
- Allow learning rate to be adjusted 


## Making BP better  


- Sequential update works well for large and highly redundant data sets.  
- Maximization of info content
    - use examples with largest error
    - use examples radically different from those previously used
    - shuffle input order
    - present more difficult inputs
- Activation function: Usually learning is faster with antisymmetric activation functions. $$\phi(-v) = - \phi(v)$$. e.g. $$\phi(v) = a \tanh(bv)$$  
- Target values, values within the range of the sigmoid activation function.  
- Normalize the inputs
- Random Initialization
- Learning from hints
- Learning rates
    - hidden layer's rates should be higher than output layer
    - small rates for neurons with many inputs


# Optimization Problem  


Derive the update rule using Taylor series expansion.  
\\[\mathcal{E}_{av}(w(n)+\Delta w(n))\\]  


## First order  


\\[\Delta w(n) = - \eta g(n)\\]  


## Second order  


Newton's method (quadratic approx)  


\\[\Delta w^*(n) = - \mathbf{H}^{-1} g(n)\\]  


Limitations:  


- expensive computation to Hessian  
- Hessian needs to be nonsingular, which is rare
- convergence is not guaranteed if the cost function is non-quadratic  


## Conjugate gradient method


Conjugate gradient method overcomes the above problems.  


**Conjugate vectors**: Given an matrix A, nonzero vectors $$s(0),s(1),...,s(W-1)$$ is A-conjugate if  
\\[s^T(n)AS(j)=0 \mbox{ for all } n \ne j\\]  


Square root of the matrix A is: $$A=A^{1/2} A^{1/2}$$.  
And $$A^{1/2} = (A^{1/2})^T$$  


Given any conjugate vector $$x_i$$ and $$x_j$$,  
\\[x_i^T A x_j=\left(A^{1/2} x_i\right)^T A^{1/2} x_j = 0\\]  


So transform any (nonzero) vector $$x_i$$ into:  
\\[v_i = A^{1/2} x_i\\]  
Results in vectors $$v_i$$ that are mutually orthogonal. They can form a basis to span the vector space.


### Conjugate Direction Method  


For a quadratic error function $$f(x)$$  
\\[x(n+1)=x(n)+\eta(n)s(n), n=0,1,...,W-1\\]  
where $$x(0)$$ is an arbitrary initial vector and $$\eta(n)$$ is defined by  
\\[f(x(n)+\eta(n)s(n)) = \min\limits_{\eta} f(x(n)+\eta(n)s(n))\\]  


Line search:  
\\[\eta(n) = - \frac{s^T(n)A \epsilon(n)}{s^T(n)A s(n)}\\]  
where $$\epsilon(n) = x(n) -x^*$$ is the error vector.  
But $$A \epsilon(n) = A x(n)- b$$  


For each iteration:  
\\[x(n+1) = \arg \min\limits_{x \in \mathcal{D}_n} f(x)\\]  
where D is spanned by the A-conjugate vectors.  


### Conjugate Gradient Method


Residual:  
\\[r(n)=b-Ax(n)\\]  
Recursive step to get conjugate vectors:  
\\[s(n) = r(n)+\beta(n) s(n-1), n=0,1,...,W-1\\]  
where scaling factor $$\beta(n)$$ is determined as  
\\[\beta(n) = - \frac{s^T(n-1)A r(n)}{s^T(n-1)A s(n)}\\]  


We need to know A. But by using Polak-Ribiere formula and Fletcher-Reeves formula, we can evaluate $$\beta(n)$$ based only on the resuduals.  


Use line search to estimate $$\eta(n)$$, can be achieved by inverse parabolic approx  


- Bracketing phase
- Sectioning phase
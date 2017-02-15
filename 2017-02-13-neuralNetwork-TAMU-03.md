---
layout: post
title: Neural Network TAMU 03 Single-Layer Perceptrons      
category: machine-learning

---


# Adapative Filter  


Training set (input/output pair):  
\\[\mathcal{T}: \{\textbf{x}(i),d(i); i = 1,2,...n\}\\]  
$$\textbf{x}(i) = [x_1(i),x_2(i),...,x_m(i)]^T$$  


- Filtering process: generation of output based on the input: $$y(i)=\textbf{x}^T(i)\textbf{w}$$  
- Adapative process: automatic adjustment of weights to reduce error: $$e(i) = d(i) - y(i)$$  


optimal solution: minimize the cost function $$\mathcal{E}(\textbf{w})$$ with respect to the weight vector.  


## Steepest Descent  


Define gradient vector $$\nabla \mathcal{E}(\textbf{w})$$ as $$g$$.  
\\[\textbf{w}(n+1)=\textbf{w}(n)-\eta g(n)\\]   


## Newton's Method  


An extension of steepest descent, where the second-order term in Taylor series expansion is used.  
Faster than steepest descent  
Need to satisfy certain conditions such as the Hessian matrix $$\nabla^2\mathcal{E}(\textbf{w})$$ being positive definite (for an arbitary $$\textbf{x}$$, $$\textbf{x}^T H \textbf{x} >0$$)  


Finally:  
\\[\textbf{w} = (X^TX)^{-1}X^T d\\]


- $$X$$ does not need to be square matrix.
- output is linear
- no iteration needed


## Least-Mean-Square Algorithm


The weight update is done with only one $$(x_i,d_i)$$ pair.  


Good for many small changes.


- like a low-pass filter
- simple, model-independent, robust  
- follow stocastical direction of steepest descent
- slow convergence
- sensitive to the input correlation matrix's condition number  
- converge when $$0 \lt \eta \lt 2/\lambda_{max}$$  


Can be improved by adapting a time-varying learning rate  


# Perceptron  


\\[v=\sum_{i=1}^{m} w_i x_i + b\\]  
\\[y = \phi(v) = \left\{ \begin{array}{cc} 1 & \mbox{ if } v \gt 0 \\\\ 0 & \mbox{ if } v \le 0 \end{array}\right. \\]  


It's impossible for a single unit to represent XOR or EQUIV  


Perceptron learning rule:  
\\[w(n+1) =w(n) +\eta (n) e(n) x(n)\\]  
where error $$e(n) = d(n) - y(n)$$  


The weight vector will tilt to fit the new input x. The perpendicular decision boundary will change.  


## Perceptron Convergence Theorem  


\\[\frac{n^2\alpha^2}{\vert \vert w_0 \vert \vert^2} \le \vert \vert w(n+1) \vert \vert ^2 \le n\beta \\]  


Thus, the number of iterations $$n$$ cannot grow beyond a certain $$n_{max}$$, where all inputs will be correctly classified:  
\\[n_{max} = \frac{\beta \vert \vert w_0 \vert \vert^2}{\alpha^2}\\]
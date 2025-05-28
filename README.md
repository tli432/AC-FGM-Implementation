# A New Convex Optimizer: AC-FGM
This repositorary contains implementation of several first-order methods for convex optimization, including the recently proposed Auto-Conditioned Fast Gradient Method (AC-FGM) presented in the companion paper https://arxiv.org/abs/2310.10082. (Recently updated: 05/27/2025)

# First-Order Methods 
The methods implemented include 
1. AC-FGM
2. Naive Gradient Descent 
3. Accelerated Gradient Descent
4. Adaptive Gradient Descent 
5. Nesterov's Universal Primal Gradient Method 
6. Nesterov's Universal Fast Gradient Method
7. FISTA

# Problem Instances 
We test the performance of the implemented optimizers on several problems instances (implemented as classes in the filder "problems")
1. Randomly Generated Quadratic Programming Problems
2. Quadratic Programming with Lasso 
3. Sparse Logistic Regression
4. Square Root Lasso 

# Updates as of 05/27/2025
We made the following updates:
1. Added FISTA with line search (Beck and Teboulle, 2009) in the method class.
2. Replaced the NS-AGD results with FISTA in all experiment sets.
3. Added the Adaptive Restart scheme to AC-FGM and FISTA, and report more experimental results on these methods.

# Reported Results
The results shown in the paper can be found in the notebook "experiments_in_the_paper.ipynb".
Extra experiments on sparse logistic regression can be found in the notebook "extra_experiments_logistic_regression.ipynb".
Experiments on AC-FGM and FISTA with adaptive restart can be found in the notebook "experiments_adaptive_restart.ipynb".

# Sample Usage
See main.py

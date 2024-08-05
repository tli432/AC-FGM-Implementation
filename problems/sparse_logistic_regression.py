import numpy as np
from scipy import sparse

class Sparse_Logistic_Regression:
    """
    This is a class for creating logistic regression instances (with L1 regularizer) with input data.
    The instance can be used as the input problem of the First_Order_Methods
    """
    def __init__(self, A, b, c=0.005):
        """
        A, b: input design matrix and response variable
        c: size of the regularizer
        """
        self.A = A
        self.b = b
        self.c = c
        b1 = sparse.diags(b)
        self.n, self.d = A.shape
        self.K = - b1 @ A
        self.gamma = c * np.linalg.norm(A.T @ b, ord=np.inf)
        _, s, _ = sparse.linalg.svds(self.K.T @ self.K, k=1)
        self.L = 1/4 * s[0]
    
    def f(self, x):
        y = self.K @ x
        f = np.sum(np.log(1 + np.exp(y)))
        return f

    def objective(self, x):
        y = self.K @ x
        f = np.sum(np.log(1 + np.exp(y)))
        h = self.gamma * np.linalg.norm(x, ord=1)
        return f + h

    def gradient(self, x):
        y = self.K @ x
        grad_f_tilde = (np.exp(y)/[1+np.exp(y)]).reshape(-1)
        grad_f = self.K.T @ grad_f_tilde
        return grad_f
    
    def first_order_oracle(self, x):
        y = self.K @ x
        f = np.sum(np.log(1 + np.exp(y)))
        h = self.gamma * np.linalg.norm(x, ord=1)
        grad_f_tilde = (np.exp(y)/[1+np.exp(y)]).reshape(-1)
        grad_f = self.K.T @ grad_f_tilde
        return f, f+h, grad_f
    
    def prox_mapping(self, x, g, gamma_scaled):
        z = np.sign(x-g) * np.max([np.abs(x-g)-gamma_scaled, np.zeros(x.shape[0])], axis =0)
        return z
    
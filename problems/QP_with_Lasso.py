import numpy as np
from scipy import sparse


class QP_with_Lasso:
    """
    This is a class for creating QP instances (with L1 regularizer) with input data.
    The instance can be used as the input problem of the First_Order_Methods
    """
    def __init__(self, A, b, c):
        """
        A, b: input design matrix and response variable
        c: size of the regularizer
        """
        self.A = A
        self.b = b
        self.n, self.d = self.A.shape
        _, s, _ = sparse.linalg.svds(self.A.T @ self.A, k=1)
        self.L = 2 * s[0]/self.n
        self.gamma = c * max(self.A.T @ self.b)/self.n

    def f(self, x):
        err = self.A @ x-self.b
        return (err @ err)/self.n
    
    def objective(self, x):
        return self.f(x) + self.gamma * np.linalg.norm(x, ord=1)
    
    def gradient(self, x):
        return 2 * (self.A.T @ (self.A @ x - self.b))/self.n
    
    def first_order_oracle(self, x):
        err = self.A @ x-self.b
        f = (err @ err)/self.n
        h = self.gamma * np.linalg.norm(x, ord=1)
        g = 2*(self.A.T @ err)/self.n
        return f, f+h, g

    def prox_mapping(self, x, g, gamma):
        z = np.sign(x-g) * np.max([np.abs(x-g)-gamma, np.zeros(x.shape[0])], axis =0)
        return z
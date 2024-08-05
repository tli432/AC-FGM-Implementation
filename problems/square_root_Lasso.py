import numpy as np
from scipy.stats import norm

class Square_Root_Lasso:
    """
    This is a class for creating square root Lasso instances with input data.
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
        self.gamma = c*norm.ppf(1-0.01/self.d)/np.sqrt(self.n)

    def f(self, x):
        return np.linalg.norm(self.A @ x-self.b, ord=2)/np.sqrt(self.n)
    
    def objective(self, x):
        return self.f(x) + self.gamma * np.linalg.norm(x, ord=1)

    def gradient(self, x):
        return 1/self.f(x)*(self.A.T @ (self.A @ x - self.b))/self.n

    def first_order_oracle(self, x):
        err = self.A @ x-self.b
        f = np.linalg.norm(err, ord=2)/np.sqrt(self.n)
        h = self.gamma * np.linalg.norm(x, ord=1)
        g = 1/f*(self.A.T @ err)/self.n
        return f, f+h, g

    def prox_mapping(self, x, g, gamma):
        z = np.sign(x-g) * np.max([np.abs(x-g)-gamma, np.zeros(x.shape[0])], axis =0)
        return z
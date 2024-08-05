import numpy as np

class Random_Logistic_Regression:
    """
    This is a class for creating random logistic regression instances.
    The instance can be used as the input problem of the First_Order_Methods
    """
    def __init__(self, m, n, c=0.0, seed=33):
        """
        (m, n): the dimension of the matrix A
        c: size of the regularizer
        seed: the random seed for reproducing same results 
        """
        np.random.seed(seed=seed)
        self.A = np.random.rand(m, n)
        self.b = (np.random.rand(m) > 0.5) * 2 - 1
        self.K = - self.A * self.b.reshape(-1, 1)
        self.n, self.d = self.A.shape
        self.gamma = c * np.linalg.norm(self.A.T @ self.b, ord=np.inf)
        self.L = np.linalg.norm(self.K.T @ self.K, ord=2)/4

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

    def prox_mapping(self, x, g, gamma):
        z = np.sign(x-g) * np.max([np.abs(x-g)-gamma, np.zeros(x.shape[0])], axis =0)
        return z
    

import numpy as np

class Random_Large_QP:
    """
    This is a class for creating random large scale QP instances.
    The instance can be used as the input problem of the First_Order_Methods
    """
    def __init__(self, m, n, seed=33):
        """
        (m, n): the dimension of the matrix A
        seed: the random seed for reproducing same results 
        """
        np.random.seed(seed=seed)
        self.A = np.random.rand(m, n)
        self.n = m
        self.d = n  
        x_star = np.random.normal(size=n) 
        x_star /= np.linalg.norm(x_star)
        self.x_star = x_star
        self.b = np.dot(self.A, x_star)
        self.L = 2 * np.linalg.norm(np.dot(self.A.T, self.A), ord=2)/self.n
        self.f_star = 0
        self.gamma = 0
        
    def objective(self, x):
        return np.dot(np.dot(self.A, x)-self.b, np.dot(self.A, x)-self.b)/self.n

    def f(self, x):
        return self.objective(x)
    
    def gradient(self, x):
        return 2 * np.dot(self.A.T, np.dot(self.A, x)-self.b)/self.n
    
    def first_order_oracle(self, x):
        err = self.A @ x-self.b
        f = (err @ err)/self.n
        g = 2*(self.A.T @ err)/self.n
        return f, f, g

    def prox_mapping(self, x, g, gamma=0):
        return x - g
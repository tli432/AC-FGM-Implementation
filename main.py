import numpy as np
from first_order_methods import First_Order_Methods
from libsvmdata import fetch_libsvm
from problems.sparse_logistic_regression import Sparse_Logistic_Regression
import pickle


def main():
    dataset = "a1a" # specify the dataset
    c = 0.001 # scale of the regularizer
    A, b = fetch_libsvm(dataset) # load the dataset
    slr = Sparse_Logistic_Regression(A, b, c) # construct the problem instance
    fom = First_Order_Methods(slr) # call the optimizer to run AC-FGM
    results_AC_FGM = fom.universal_AC_FGM(k=10000, beta = 0.184, alpha=0.1, initial_line_search=False)
    with open('tmp.pkl', 'wb') as f: 
        pickle.dump(results_AC_FGM, f)

if __name__ == '__main__':
    main()


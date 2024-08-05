import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import time


class First_Order_Methods:
    """
    A general class contatining several optimization solvers, including AC-FGM
    """
    def __init__(self, prob, x_0=None):
        """
        prob: Read the optimization problem and its parameters
            A problem should contains function value, gradients, a first order oracle, and the prox-mapping subproblem
            see an example in sparse_logisitic_regression.py
        x_0: The initialization of the optimizer, if no x_0, set it as the origin
        """
        self.prob = prob
        self.d = prob.d
        if x_0:
            self.x_0 = x_0
        else:
            self.x_0 = np.zeros(self.d)

    def GD(self, k, L, gamma=None, x_0=None):
        """
        Naive gradient descent with constant stepsize

        Input: 
        k: number of iterations
        L: the Lipschitz constant for smooth problems
        gamma: scale of the regularizer, if applicable, usually set as prob.gamma
        x_0: the initialized search point

        Output: function values in all iterations, output solution, cumulative number of oraclue calls, total run time
        """
        print("GD: k=%d"%(k))
        T_1 = time.clock()
        if gamma is None:
            gamma = self.prob.gamma
        if x_0 is not None:
            x = x_0.copy()
        else:
            x = self.x_0.copy()
        _, obj, g = self.prob.first_order_oracle(x)
        values = [obj]
        oracles = 1
        num_oracles = [1]
        
        for t in range(1, k+1):
            eta = 2/L
            x = self.prob.prox_mapping(x, eta * g, eta * gamma)
            _, obj, g = self.prob.first_order_oracle(x)
            oracles += 1
            values.append(obj)
            num_oracles.append(oracles)
        
        T_2 = time.clock()
        print("Done: %.3f seconds"%(T_2 - T_1))
        print("-----------------------------------")
        
        return np.array(values), x, np.array(num_oracles), T_2 - T_1

    def nesterovAGD(self, k, L, gamma=None, x_0=None):
        """
        Nonadaptive accelerated gradient descent with constant stepsize

        Input: 
        k: number of iterations
        L: the Lipschitz constant for smooth problems
        gamma: scale of the regularizer, if applicable, usually set as prob.gamma
        x_0: the initialized search point

        Output: function values in all iterations, output solution, cumulative number of oraclue calls, total run time
        """
        print("NS-AGD: k=%d"%(k))
        T_1 = time.clock()
        if gamma is None:
            gamma = self.prob.gamma
        if x_0 is not None:
            x = x_0.copy()
        else:
            x = self.x_0.copy()
        x_bar = x.copy()
        x_und = x.copy()
        obj = self.prob.objective(x_bar)
        values = [obj]
        oracles = 1
        num_oracles = [1]
        
        for t in range(1, k+1):
            q = 2/(t+1)
            alpha = 2/(t+1)
            eta = t/2/L
            x_und = (1-q)*x_bar + q*x
            x = self.prob.prox_mapping(x, eta * self.prob.gradient(x_und), eta * gamma)
            x_bar = (1-alpha)*x_bar + alpha*x
            value = self.prob.objective(x_bar)
            values.append(value)
            oracles += 1
            num_oracles.append(oracles)
        
        T_2 = time.clock()
        print("Done: %.3f seconds"%(T_2 - T_1))
        print("-----------------------------------")
        
        return np.array(values), x_bar, np.array(num_oracles), T_2 - T_1

    def universal_AC_FGM(self, k, L_0=None, gamma=None, beta=0.184, alpha=0.1, x_0=None, eps=0, initial_line_search=True):
        """
        AC-FGM method (universal version)

        Input: 
        k: number of iterations
        L_0: initial guess of the Lipschitz smooth constant. If L_0 = None, the method will use a local L_0 around x_0
        gamma: scale of the regularizer, if applicable, usually set as prob.gamma
        beta: algorithm parameter, can be chosen between 0 and 1. The default value of 0.184 comes from the paper
        alpha: algorithm parameter, can be chosen between 0 and 1. The default value is 0.1
        x_0: the initialized search point
        eps: pre-set accuracy for universal objectives. If the function is smooth, set eps = 0.
        initial line search: binary variable, decide whether to do an initial line search step.

        Output: function values in all iterations, output solution, cumulative number of oraclue calls, total run time
        """
        if eps > 0:
            print("AC-FGM: k=%d, alpha=%.3f, beta=%.3f, eps=1e%d"%(k, alpha, beta, np.log10(eps)))
        else:
            print("AC-FGM: k=%d, alpha=%.3f, beta=%.3f"%(k, alpha, beta))
        T_1 = time.clock()
        if gamma is None:
            gamma = self.prob.gamma
        if x_0 is not None:
            x = x_0.copy()
        else:
            x = self.x_0.copy()
        z = x.copy()
        y = x.copy()
        f, obj, g = self.prob.first_order_oracle(x)
        values = [obj]
        oracles = 1
        num_oracles = [1]
        
        if L_0 is None:
            x_eps = x - np.ones(self.prob.d) * 0.1
            g_eps = self.prob.gradient(x_eps)
            L_0 = np.linalg.norm(g_eps-g)/np.linalg.norm(x_eps-x)
            oracles += 1

        if initial_line_search == True:
            # First iteration: utilizing initial line search to determine eta_1
            for i in range(200):
                L_new = 1.5**i * L_0 / 4
                eta = 1 / 2.5 / L_new
                x_new = self.prob.prox_mapping(x, eta*g, eta*gamma)
                f_new, obj_new, g_new = self.prob.first_order_oracle(x_new)
                oracles += 1
                tmp1 = (np.linalg.norm(g_new-g))**2/2/L_new
                tmp2 = L_new * (np.linalg.norm(x_new-x))**2 / 2 + eps/4
                if tmp1 <= tmp2:
                    print("Number of calls used for the initial linesearch: %d"%(i+1))
                    break
            
            if i == 199:
                print("Error: Need to rerun the linesearch.")
                return None
            
        else:
            # If no line search: determine eta_1 based on L_0
            L_new = L_0
            eta = 1 / 2.5 /L_new
            x_new = self.prob.prox_mapping(x, eta*g, eta*gamma)
            f_new, obj_new, g_new = self.prob.first_order_oracle(x_new)
            oracles += 1

        x = x_new.copy()  
        z = x_new.copy()  
        y = y   
        L_e = L_new
        f = f_new
        g = g_new
        values.append(obj_new)
        num_oracles.append(oracles)

        # Update rule of each iterations
        for t in range(2, k+1):
            if t == 2:
                eta = np.min([(1-beta)*eta, 1/4/L_e])
                tau_old = 0
                tau = 1
            if t >= 2:
                if L_e > 0:
                    eta = np.min([4/3*eta, (tau_old+1)/tau*eta, tau/4/L_e])
                tau_old = tau
                tau = tau + 2*(1-alpha) * eta * L_e/tau + alpha/2
            
            z = self.prob.prox_mapping(y, eta*g, eta*gamma)
            y = (1-beta)*y + beta*z
            x_new = (z + tau * x)/(1+tau)
            f_new, obj_new, g_new = self.prob.first_order_oracle(x_new)
            oracles += 1
            if np.linalg.norm(g_new - g) == 0:
                L_e = 0
            else:
                L_e = np.linalg.norm(g_new - g)**2/(2*(f-f_new-np.dot(g_new, x-x_new))+ eps/tau)
                if L_e < 0:
                    L_e = 0
            obj = obj_new
            f = f_new
            g = g_new
            x = x_new
            values.append(obj)
            num_oracles.append(oracles)

        T_2 = time.clock()
        print("Done: %.3f seconds"%(T_2 - T_1))
        print("-----------------------------------")

        return np.array(values), x, np.array(num_oracles), T_2-T_1

    def adaptiveGD(self, k, L_0=None, gamma=None, x_0=None):
        """
        Adaptive Gradient Descent (AdGD) for smooth objective (Malitsky and Mishchenko (2023))

        Input: 
        k: number of iterations
        L_0: initial guess of the Lipschitz smooth constant. If L_0 = None, the method will use a local L_0 around x_0
        gamma: scale of the regularizer, if applicable, usually set as prob.gamma
        x_0: the initialized search point

        Output: function values in all iterations, output solution, cumulative number of oraclue calls, total run time
        """
        print("AdaGD: k=%d"%k)
        T_1 = time.clock()
        if gamma is None:
            gamma = self.prob.gamma
        if x_0 is not None:
            x = x_0.copy()
        else:
            x = self.x_0.copy()
        _, obj, g = self.prob.first_order_oracle(x)
        values = [obj]
        oracles = 1
        num_oracles = [1]

        if L_0 is None:
            x_eps = x - np.ones(self.prob.d) * 0.1
            g_eps = self.prob.gradient(x_eps)
            L_0 = np.linalg.norm(g_eps-g)/np.linalg.norm(x_eps-x)/4
            oracles += 1
        
        # Do an initial line search to determine eta_1
        for i in range(200):
            L_new = 2**i * L_0
            eta = 1 / 3 / L_new
            x_new = self.prob.prox_mapping(x, eta*g, eta*gamma)
            g_new = self.prob.gradient(x_new)
            oracles += 1
            tmp1 = (np.linalg.norm(g_new-g))**2/2/L_new
            tmp2 = L_new * (np.linalg.norm(x_new-x))**2 / 2
            if tmp1 <= tmp2:
                print("Number of calls used for the initial linesearch: %d"%(i+1))
                break
        
        if i == 200:
            print("Need to rerun the linesearch.")
            return None
        
        L_e = L_new
        theta = 0
        x_sum = 0
        eta_sum = 0

        # Update fule for each iterations
        for t in range(2, k+1):
            _, obj_new, g_new = self.prob.first_order_oracle(x_new)
            oracles += 1
            if np.linalg.norm(x_new-x) == 0:
                L_e = 0
            else:
                L_e = np.linalg.norm(g_new-g)/np.linalg.norm(x_new-x)
            if L_e > 0:
                eta_new = np.min([eta * np.sqrt(1+theta), 1/L_e/np.sqrt(2)])
            else:
                eta_new = eta * np.sqrt(1+theta)

            theta = eta_new/eta
            eta = eta_new
            g = g_new
            x = x_new
            obj = obj_new
            x_sum += eta * x
            eta_sum += eta
            values.append(obj)
            num_oracles.append(oracles)
            x_new = self.prob.prox_mapping(x, eta * g, eta * gamma)

        T_2 = time.clock()
        print("Done: %.3f seconds"%(T_2 - T_1))
        print("-----------------------------------")

        return np.array(values), x, np.array(num_oracles), T_2-T_1

    def universalPGM(self, k, L_0=None, gamma=None, x_0=None, eps=0):
        """
        Primal gradient method (NS-PGM) for universal objective (Nesterov (2015))

        Input: 
        k: number of iterations
        L_0: initial guess of the Lipschitz smooth constant. If L_0 = None, the method will use a local L_0 around x_0
        gamma: scale of the regularizer, if applicable, usually set as prob.gamma
        x_0: the initialized search point

        Output: function values in all iterations, output solution, cumulative number of oraclue calls, total run time
        """
        if eps > 0:
            print("NS-PGM: k=%d, eps=1e%d"%(k, np.log10(eps)))
        else:
            print("NS-PGM: k=%d"%(k))
        T_1 = time.clock()
        if gamma is None:
            gamma = self.prob.gamma
        if x_0 is not None:
            x = x_0.copy()
        else:
            x = self.x_0.copy()
        f, obj, g = self.prob.first_order_oracle(x)
        values = [obj]
        oracles = 1
        num_oracles = [1]
        if L_0 is None:
            x_eps = x - np.ones(self.prob.d) * 0.1
            g_eps = self.prob.gradient(x_eps)
            L_0 = np.linalg.norm(g_eps-g)/np.linalg.norm(x_eps-x)/4
            oracles += 1

        L = L_0
        sum_L = 0
        sum_x = 0

        for t in range(1, k+1):
            for i in range(100):
                L_new = 2**i * L
                x_new = self.prob.prox_mapping(x, 1/L_new*g, 1/L_new*gamma)
                f_new = self.prob.f(x_new)
                oracles += 1
                tmp1 = f_new - f - np.dot(g, x_new-x)
                tmp2 = L_new/2 * (np.linalg.norm(x_new-x))**2 + eps/2
                if tmp1 <= tmp2:
                    break
            f = f_new
            x = x_new
            L = L_new/2
            sum_L += 1/L
            sum_x += 1/L * x
            x_hat = sum_x/sum_L
            f, obj, g = self.prob.first_order_oracle(x)
            values.append(obj)
            num_oracles.append(oracles)
        T_2 = time.clock()
        print("Done: %.3f seconds"%(T_2 - T_1))
        print("-----------------------------------")

        return np.array(values), x, np.array(num_oracles), T_2 - T_1
    
    def UniversalFGM(self, k, L_0=None, gamma=None, x_0=None, eps=0):
        """
        Universal fast gradient method (NS-FGM) for universal objective (Nesterov (2015))

        Input: 
        k: number of iterations
        L_0: initial guess of the Lipschitz smooth constant. If L_0 = None, the method will use a local L_0 around x_0
        gamma: scale of the regularizer, if applicable, usually set as prob.gamma
        x_0: the initialized search point

        Output: function values in all iterations, output solution, cumulative number of oraclue calls, total run time
        """
        if eps > 0:
            print("NS-FGM: k=%d, eps=1e%d"%(k, np.log10(eps)))
        else:
            print("NS-FGM: k=%d"%(k))
        T_1 = time.clock()
        if gamma is None:
            gamma = self.prob.gamma
        if not x_0:
            x_0 = self.x_0.copy()
        x = x_0.copy()
        f, obj, g = self.prob.first_order_oracle(x)
        values = [obj]
        oracles = 1
        num_oracles = [1]
        y = x.copy()
        A = 0
        dual_phi = np.zeros(self.d)
        const_psi = 0
        if L_0 is None:
            x_eps = x - np.ones(self.prob.d) * 0.1
            g_eps = self.prob.gradient(x_eps)
            L_0 = np.linalg.norm(g_eps-g)/np.linalg.norm(x_eps-x)/4
            oracles += 1
        L = L_0

        for t in range(1, k+1):
            v = self.prob.prox_mapping(x_0, dual_phi, const_psi * gamma)
            for i in range(1000):
                L_new = 2**i * L
                a_new = (1+np.sqrt(1+4*L_new*A))/2/L_new
                A_new = A + a_new
                tau = a_new / A_new
                x_new = tau * v + (1-tau) * y
                f_x, _, g_new = self.prob.first_order_oracle(x_new)
                x_hat_new = self.prob.prox_mapping(v, a_new * g_new, a_new * gamma)
                y_new = tau * x_hat_new + (1-tau) * y
                f_y = self.prob.f(y_new)
                oracles += 2
                tmp1 = f_y - f_x - np.dot(g_new, y_new - x_new)
                tmp2 = L_new/2 * (np.linalg.norm(y_new-x_new))**2 + eps * tau/2
                if tmp1 <= tmp2:
                    break
            x = x_new
            y = y_new
            A = A_new
            L = L_new/2
            a = a_new
            dual_phi += a * g_new
            const_psi += a
            obj = self.prob.objective(y)
            values.append(obj)
            num_oracles.append(oracles)
        T_2 = time.clock()
        print("Done: %.3f seconds"%(T_2 - T_1))
        print("-----------------------------------")
        
        return np.array(values), y, np.array(num_oracles), T_2 - T_1
    
    def test_all_methods_nonsmooth(self, k, beta=0.184, L_0=None, x_0=None, eps=1e-8, initial_line_search=True, draw_performance_plot=False, dataset="",  right_lim=15000):
        """
        A simple function for testing the performance of different optimizer in solving a nonsmooth problem

        Input: 
        k: number of iterations for each method
        beta: the beta parameter for AC-FGM
        L_0: initial guess of the Lipschitz smooth constant. If L_0 = None, the method will use a local L_0 around x_0
        x_0: the initialized search point
        eps: preset target accuracy for universal objective
        initial_line_search: whether performing initial line search for AC-FGM
        draw_performance_plot: whether draw performance graph
        dataset: name of the dateset (used in the plot name)
        right_lim: right limit for the x axis of the plots (number of iterations)

        Output: output parameters of each of the optimizers, the estimated optimal function value
        """
        results_AC_FGM = self.universal_AC_FGM(k=k, beta = beta, x_0=x_0, alpha=0.0, eps=eps, gamma=self.prob.gamma, initial_line_search=initial_line_search)
        results_AC_FGM_01 = self.universal_AC_FGM(k=k, beta = beta, x_0=x_0, alpha=0.1, eps=eps, gamma=self.prob.gamma, initial_line_search=initial_line_search)
        results_AC_FGM_02 = self.universal_AC_FGM(k=k, beta = beta, x_0=x_0, alpha=0.5, eps=eps, gamma=self.prob.gamma, initial_line_search=initial_line_search)
        results_NS_PGM = self.universalPGM(k=k, gamma=self.prob.gamma, eps=eps)
        results_NS_FGM = self.UniversalFGM(k=k, gamma=self.prob.gamma, eps=eps)
        f_star = min([min(results_AC_FGM[0]), min(results_AC_FGM_01[0]), min(results_AC_FGM_02[0]), min(results_NS_PGM[0]), min(results_NS_FGM[0])])
        
        if draw_performance_plot == True:
            plt.figure(figsize=(9,6.5),dpi=150)
            plt.plot(results_AC_FGM_02[0]-f_star, "b", linewidth=3)
            plt.plot(results_AC_FGM_01[0]-f_star, "y", linewidth=3)
            plt.plot(results_AC_FGM[0]-f_star, "m", linewidth=3)
            plt.plot(results_NS_PGM[0]-f_star, ":r", linewidth=3)
            plt.plot(results_NS_FGM[0]-f_star, "--k", linewidth=3)
            plt.legend(["AC-FGM:0.5", "AC-FGM:0.1", "AC-FGM:0.0", "NS-PGM", "NS-FGM"], fontsize=15)
            plt.yscale("log")
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.ylabel("Error", fontsize=15)
            plt.xlabel("iterations", fontsize=15)
            plt.grid(linestyle= '--')
            plt.title(dataset, fontsize=22)
            plt.show()

            plt.figure(figsize=(9,6.5),dpi=150)
            plt.plot(results_AC_FGM_02[2], results_AC_FGM_02[0]-f_star, "b", linewidth=3)
            plt.plot(results_AC_FGM_01[2], results_AC_FGM_01[0]-f_star, "y", linewidth=3)
            plt.plot(results_AC_FGM[2], results_AC_FGM[0]-f_star, "m", linewidth=3)
            plt.plot(results_NS_PGM[2], results_NS_PGM[0]-f_star, ":r", linewidth=3)
            plt.plot(results_NS_FGM[2], results_NS_FGM[0]-f_star, "--k", linewidth=3)
            plt.legend(["AC-FGM:0.5", "AC-FGM:0.1", "AC-FGM:0.0", "NS-PGM", "NS-FGM"], fontsize=15)
            plt.yscale("log")
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.gca().set_xlim(right=right_lim)
            plt.ylabel("Error", fontsize=15)
            plt.xlabel("oracle calls", fontsize=15)
            plt.grid(linestyle= '--')
            plt.title(dataset, fontsize=22)
            plt.show()

        return [results_AC_FGM, results_AC_FGM_01, results_AC_FGM_02, results_NS_PGM, results_NS_FGM], f_star
    

    def test_all_methods_smooth(self, k, beta=0.184, L_0=None, x_0=None, initial_line_search=True, draw_performance_plot=False, dataset="", f_star=None, right_lim=15000, save_fig=False, save_title="plot", xsize = 15):
        """
        A simple function for testing the performance of different optimizer in solving a smooth problem

        Input: 
        k: number of iterations for each method
        beta: the beta parameter for AC-FGM
        L_0: initial guess of the Lipschitz smooth constant. If L_0 = None, the method will use a local L_0 around x_0
        x_0: the initialized search point
        initial_line_search: whether performing initial line search for AC-FGM
        draw_performance_plot: whether draw performance graph
        dataset: name of the dateset (used in the plot name)
        f_star: if the optimal value is known, the user can set it in advance, e.g., f_star = 0 when solving a linear system
        right_lim: right limit for the x axis of the plots (number of iterations)

        Output: output parameters of each of the optimizers, the estimated optimal function value
        """
        results_AC_FGM = self.universal_AC_FGM(k=k, beta = beta, x_0=x_0, alpha=0.0, eps=0, gamma=self.prob.gamma, initial_line_search=initial_line_search)
        results_AC_FGM_01 = self.universal_AC_FGM(k=k, beta = beta, x_0=x_0, alpha=0.1, eps=0, gamma=self.prob.gamma, initial_line_search=initial_line_search)
        results_AC_FGM_02 = self.universal_AC_FGM(k=k, beta = beta, x_0=x_0, alpha=0.5, eps=0, gamma=self.prob.gamma, initial_line_search=initial_line_search)
        results_adaptiveGD = self.adaptiveGD(k=k, gamma=self.prob.gamma)
        results_NS_FGM = self.UniversalFGM(k=k, gamma=self.prob.gamma, eps=0)
        results_NS_AGD = self.nesterovAGD(k=k, gamma=self.prob.gamma, L=self.prob.L)
        if f_star == None:
            f_star = min([min(results_AC_FGM[0]), min(results_AC_FGM_01[0]), min(results_AC_FGM_02[0]), min(results_adaptiveGD[0]), min(results_NS_FGM[0]), min(results_NS_AGD[0])])

        if draw_performance_plot == True:
            plt.figure(figsize=(9,6.5),dpi=150)
            plt.plot(results_AC_FGM_02[0]-f_star, "b", linewidth=3)
            plt.plot(results_AC_FGM_01[0]-f_star, "y", linewidth=3)
            plt.plot(results_AC_FGM[0]-f_star, "m", linewidth=3)
            plt.plot(results_NS_AGD[0]-f_star, ":r", linewidth=3)
            plt.plot(results_adaptiveGD[0]-f_star, "-.g", linewidth=3)
            plt.plot(results_NS_FGM[0]-f_star, "--k", linewidth=3)
            plt.legend(["AC-FGM:0.5", "AC-FGM:0.1", "AC-FGM:0.0", "NS-AGD", "AdGD", "NS-FGM"], fontsize=15)
            plt.yscale("log")
            plt.xticks(fontsize=xsize)
            plt.yticks(fontsize=15)
            plt.ylabel("Error", fontsize=15)
            plt.xlabel("Iterations", fontsize=15)
            plt.grid(linestyle= '--')
            plt.title(dataset, fontsize=22)
            if save_fig == True:
                plt.savefig("%s.png"%save_title)
            else:
                plt.show()

            plt.figure(figsize=(9,6.5),dpi=150)
            plt.plot(results_AC_FGM_02[2], results_AC_FGM_02[0]-f_star, "b", linewidth=3)
            plt.plot(results_AC_FGM_01[2], results_AC_FGM_01[0]-f_star, "y", linewidth=3)
            plt.plot(results_AC_FGM[2], results_AC_FGM[0]-f_star, "m", linewidth=3)
            plt.plot(results_NS_AGD[2], results_NS_AGD[0]-f_star, ":r", linewidth=3)
            plt.plot(results_adaptiveGD[2], results_adaptiveGD[0]-f_star, "-.g", linewidth=3)
            plt.plot(results_NS_FGM[2], results_NS_FGM[0]-f_star, "--k", linewidth=3)
            plt.legend(["AC-FGM:0.5", "AC-FGM:0.1", "AC-FGM:0.0", "NS-AGD", "AdGD", "NS-FGM"], fontsize=15)
            plt.yscale("log")
            plt.xticks(fontsize=xsize)
            plt.yticks(fontsize=15)
            plt.gca().set_xlim(right=right_lim)
            plt.ylabel("Error", fontsize=15)
            plt.xlabel("Oracle Calls", fontsize=15)
            plt.grid(linestyle= '--')
            plt.title(dataset, fontsize=22)
            if save_fig == True:
                plt.savefig("%s_o.png"%save_title)
            else:
                plt.show()

        return [results_AC_FGM, results_AC_FGM_01, results_AC_FGM_02, results_adaptiveGD, results_NS_FGM, results_NS_AGD], f_star
        

    def test_linesearch_free_methods_smooth(self, k, beta=0.184, L_0=None, x_0=None, initial_line_search=True, draw_performance_plot=False, dataset="", f_star=None, right_lim=15000, save_fig=False, save_title="plot", xsize = 15):
        """
        A simple function for testing the performance of different optimizer in solving a smooth problem

        Input: 
        k: number of iterations for each method
        beta: the beta parameter for AC-FGM
        L_0: initial guess of the Lipschitz smooth constant. If L_0 = None, the method will use a local L_0 around x_0
        x_0: the initialized search point
        initial_line_search: whether performing initial line search for AC-FGM
        draw_performance_plot: whether draw performance graph
        dataset: name of the dateset (used in the plot name)
        f_star: if the optimal value is known, the user can set it in advance, e.g., f_star = 0 when solving a linear system
        right_lim: right limit for the x axis of the plots (number of iterations)

        Output: output parameters of each of the optimizers, the estimated optimal function value
        """
        results_AC_FGM = self.universal_AC_FGM(k=k, beta = beta, x_0=x_0, alpha=0.0, eps=0, gamma=self.prob.gamma, initial_line_search=initial_line_search)
        results_AC_FGM_01 = self.universal_AC_FGM(k=k, beta = beta, x_0=x_0, alpha=0.1, eps=0, gamma=self.prob.gamma, initial_line_search=initial_line_search)
        results_AC_FGM_02 = self.universal_AC_FGM(k=k, beta = beta, x_0=x_0, alpha=0.5, eps=0, gamma=self.prob.gamma, initial_line_search=initial_line_search)
        results_adaptiveGD = self.adaptiveGD(k=k, gamma=self.prob.gamma)
        if f_star == None:
            f_star = min([min(results_AC_FGM[0]), min(results_AC_FGM_01[0]), min(results_AC_FGM_02[0]), min(results_adaptiveGD[0])])

        if draw_performance_plot == True:
            plt.figure(figsize=(9,6.5),dpi=150)
            plt.plot(results_AC_FGM_02[0]-f_star, "b", linewidth=3)
            plt.plot(results_AC_FGM_01[0]-f_star, "y", linewidth=3)
            plt.plot(results_AC_FGM[0]-f_star, "m", linewidth=3)
            plt.plot(results_adaptiveGD[0]-f_star, "-.g", linewidth=3)
            plt.legend(["AC-FGM:0.5", "AC-FGM:0.1", "AC-FGM:0.0", "AdGD"], fontsize=15)
            plt.yscale("log")
            plt.xticks(fontsize=xsize)
            plt.yticks(fontsize=15)
            plt.ylabel("Error", fontsize=15)
            plt.xlabel("Iterations", fontsize=15)
            plt.grid(linestyle= '--')
            plt.title(dataset, fontsize=22)
            if save_fig == True:
                plt.savefig("%s.png"%save_title)
            else:
                plt.show()

            plt.figure(figsize=(9,6.5),dpi=150)
            plt.plot(results_AC_FGM_02[2], results_AC_FGM_02[0]-f_star, "b", linewidth=3)
            plt.plot(results_AC_FGM_01[2], results_AC_FGM_01[0]-f_star, "y", linewidth=3)
            plt.plot(results_AC_FGM[2], results_AC_FGM[0]-f_star, "m", linewidth=3)
            plt.plot(results_adaptiveGD[2], results_adaptiveGD[0]-f_star, "-.g", linewidth=3)
            plt.legend(["AC-FGM:0.5", "AC-FGM:0.1", "AC-FGM:0.0", "AdGD"], fontsize=15)
            plt.yscale("log")
            plt.xticks(fontsize=xsize)
            plt.yticks(fontsize=15)
            plt.gca().set_xlim(right=right_lim)
            plt.ylabel("Error", fontsize=15)
            plt.xlabel("Oracle Calls", fontsize=15)
            plt.grid(linestyle= '--')
            plt.title(dataset, fontsize=22)
            if save_fig == True:
                plt.savefig("%s_o.png"%save_title)
            else:
                plt.show()

        return [results_AC_FGM, results_AC_FGM_01, results_AC_FGM_02, results_adaptiveGD], f_star


    




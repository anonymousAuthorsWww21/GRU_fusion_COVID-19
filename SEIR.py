import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import basinhopping

data_dir = "../data/Italy.csv"

# Population of selected countries:
# Italy       = 60431283
# Germany     = 82927922
# Russia      = 145934462
# Denmark     = 5792202
# Austria     = 9006398
# Switzerland = 8654622
N = 60431283
predict_range = 150
iter_num = 1
loss_weight = 0.3


class SEIRfit(object):
    def __init__(self, loss, predict_range):
        self.loss = loss
        self.predict_range = predict_range

    def load_confirmed(self):
        df = pd.read_csv(data_dir)
        dff = df["D"]
        i_0 = dff[0]
        return dff.T, i_0

    def load_removed(self):
        df = pd.read_csv(data_dir)
        dff1 = df["R"]
        dff2 = df["A"]
        dff = dff1 + dff2
        r_0 = dff[0]
        return dff.T, r_0

    def predict(self, beta, alpha, gamma, k, E_0, I_0, R_0, D_actual):
        size = len(D_actual) + predict_range
        S_0 = N*k - I_0 - R_0 - E_0

        def SEIR(t, y):
            S = y[0]
            E = y[1]
            I = y[2]
            R = y[3]
            return [-beta*S*I/N, beta*S*I/N-alpha*E, alpha*E-gamma*I, gamma*I]

        return solve_ivp(SEIR, [0, size], [S_0, E_0, I_0, R_0], t_eval=np.arange(0, size, 1))
    
    def fit(self):
        D_actual, I_0 = self.load_confirmed()
        R_actual, R_0 = self.load_removed()
        print("Start\n")
        optimal = basinhopping(loss, [0.001, 0.001, 0.001, 0.001, 1],
                               niter=iter_num,
                               minimizer_kwargs = {
                                   "method": "L-BFGS-B",
                                   "args":(D_actual, R_actual, loss_weight, I_0, R_0),
                                   "bounds":[(0.00000001, 50.), (0.00000001, 1.), (0.00000001, 1.), (0.0000000001, 1.), (0, 312)]
                               })
        print(optimal)
        beta, alpha, gamma, k, E_0 = optimal.x
        print(f" beta={beta:.8f}, alpha={alpha:.8f}, gamma={gamma:.8f}, k:{k:.8f}, E_0:{E_0:.8f}")

        prediction = self.predict(beta, alpha, gamma, k, E_0, I_0, R_0, D_actual)
        D = prediction.y[2]
        R = prediction.y[3]
        result = pd.DataFrame({'D': D, 'R': R})
        result.to_csv('./output/result_SEIR.csv', encoding='gbk')


def loss(param, D_actual, R_actual, w, I_0, R_0):
    size = len(D_actual)
    beta, alpha, gamma, k, E_0 = param
    S_0 = N*k - I_0 - R_0 - E_0

    def SEIR(t, y):
        S = y[0]
        E = y[1]
        I = y[2]
        R = y[3]
        return [-beta*S*I/N, beta*S*I/N-alpha*E, alpha*E-gamma*I, gamma*I]

    solution = solve_ivp(SEIR, [0, size], [S_0, E_0, I_0, R_0], t_eval=np.arange(0, size, 1), vectorized=True)
    loss1 = np.mean((solution.y[2] - D_actual) ** 2)
    loss2 = np.mean((solution.y[3] - R_actual) ** 2)
    return (1-w)*loss1 + w*loss2


def main():
    SEIRfitter = SEIRfit(loss, predict_range)
    SEIRfitter.fit()


if __name__ == '__main__':
    main()
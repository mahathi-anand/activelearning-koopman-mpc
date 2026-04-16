#This code has been adapted from https://github.com/i-abr/active-learning-koopman/


import numpy as np 
import pandas as pd
from system import LinearSystem
from system import VanDerPolSystem
import matplotlib.pyplot as plt

class KoopmanOperator:
    def __init__(self, system, noise=1.0):
        self.system = system
        self.noise = noise
        self.n_x = system.n_x
        self.n_u = system.n_u
        self.num_observables = system.observables(np.zeros(system.n_x))[1]
        #Parameters required for least squares regression
        self.V = np.zeros((self.num_observables, self.num_observables + self.n_u)) #Dimension N x (N+m)
        self.G = np.zeros((self.num_observables + self.n_u, self.num_observables + self.n_u)) #Dimension (N+m) x (N+m) 
        #Initialization of Koopman operator matrices
        self.Kx = np.ones((self.num_observables, self.num_observables))
        self.Ku = np.ones((self.num_observables, self.n_u))
        self. K = np.hstack((self.Kx, self.Ku)) #Dimension N x (N+m)
        self.N_data = 0 # initially no data, will be updated as we collect more data
    
    def compute_operator(self, X, U, X_plus):
        N_new = X.shape[1]
        Z = self.system.observables(X)[0]
        Z_plus = self.system.observables(X_plus)[0]
        Z_aug = np.vstack((Z, U)).T
        V_new = np.zeros_like(self.V)
        G_new = np.zeros_like(self.G)
        for i in range(N_new):
            V_new = V_new + np.outer(Z_plus[:, i], Z_aug[i, :])
            G_new = G_new + np.outer(Z_aug[i, :], Z_aug[i, :])
        V_new = V_new / N_new
        G_new = G_new / N_new
        self.V = (N_new * V_new + self.N_data * self.V) / (N_new + self.N_data)
        self.G = (N_new * G_new + self.N_data * self.G) / (N_new + self.N_data)
        self.N_data += N_new

        #Koopman operator compute
        try: 
            self.K = np.linalg.solve(self.G.T, self.V.T).T
        except np.linalg.LinAlgError:
            print("singular matrix encountered, using pseudo-inverse instead")
            self.K = self.V @ np.linalg.pinv(self.G)
        self.Kx = self.K[:, :self.num_observables]
        self.Ku = self.K[:, self.num_observables:]

        return self.Kx, self.Ku


if __name__ == "__main__":
    system = VanDerPolSystem(mu=0.0, dt=0.01)
    print(f"System: {system.system_name}, State Dimension: {system.n_x}, Control Dimension: {system.n_u}")
    koopman_operator = KoopmanOperator(system, noise=1.0)
    data = pd.read_csv('data/' + system.system_name + '.csv')
    X = data[['x1', 'x2']].values.T
    U = data[['u']].values.T
    X_plus = data[['x1_plus', 'x2_plus']].values.T
    print(f"Data Loaded: {X.shape[1]} samples... Computing Koopman Operator...")
    [A, B] = koopman_operator.compute_operator(X, U, X_plus)

    print(f"Koopman Operator Computed: A shape={A.shape}, B shape={B.shape}")
    #Koopman System
    linear_system = LinearSystem(A, B, name="VanDerPol")
    #Simulate both systems to compare trajectories
    num_steps = 2000

    rows = 2
    columns = 2
    fig, axes = plt.subplots(rows, columns, figsize=(10, 10))
    fig.suptitle('Comparison of Nonlinear and Koopman System Trajectories - ' + system.system_name)

    for u in range(rows * columns):
        ax = plt.subplot(2,2, u+1)
        x0 = np.random.uniform(-5, 5, size=(2,))
        #Simulate original system
        u_seq = np.random.uniform(-2, 2, size=(1, num_steps))
        traj_vanderpol = system.simulate(x0, u_seq.T)

        #Simulate Koopman system
        z0, _ = system.observables(x0)
        traj_koopman = linear_system.simulate(z0, u_seq.T)
        traj_koopman_ext = traj_koopman[:, :2] #Extracting the original state dimensions from the Koopman trajectory

        #Plotting trajectories  
        ax.plot(traj_vanderpol[:, 0], traj_vanderpol[:, 1], label='Van der Pol', color='blue')
        ax.plot(traj_koopman[:, 0], traj_koopman[:, 1], label='Koopman', color='blue', linestyle='dashed')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()

    plt.tight_layout()
    plt.show()

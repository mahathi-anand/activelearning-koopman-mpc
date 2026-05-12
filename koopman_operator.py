#This code has been adapted from https://github.com/i-abr/active-learning-koopman/


from imports import * 
from system import LinearSystem
from system import VanDerPolSystem

class KoopmanOperator:
    def __init__(self, system):
        self.system = system
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



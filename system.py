import numpy as np

class LinearSystem:
    def __init__(self, A, B, name = "VanDerPol"):
        self.A = A
        self.B = B
        self.n_x = A.shape[0]
        self.n_u = B.shape[1]
        self.system_name = "linear_" + name

    #system dynamics
    def dynamics(self, x, u):
        return self.A @ x + self.B @ u
    
    #simulate system
    def simulate(self, x0, u_seq):
        x = x0
        trajectory = [x]
        for u in u_seq:
            x = self.dynamics(x, u)
            trajectory.append(x)
        return np.array(trajectory)

class VanDerPolSystem:
    def __init__(self, mu=1.0, dt=0.01):
        self.n_x = 2
        self.n_u = 1
        self.mu = mu
        self.dt = dt
        self.system_name = "VanDerPol"

    #system dynamics    
    def dynamics(self, x, u):
        x1 = x[0] + self.dt * x[1]
        x2 = x[1] + self.dt * (self.mu * (1 - x[0] ** 2) * x[1] - x[0] + u[0])
        return np.array([x1, x2]).transpose()
    
    #simulate system if needed
    def simulate(self, x0, u_seq):
        x = x0
        trajectory = [x]
        for u in u_seq:
            x = self.dynamics(x, u)
            trajectory.append(x)
        return np.array(trajectory)
    
    def observables(self, x):
        obs = np.array([x[0], x[1], x[0]**2, x[1]**2, x[0]**2*x[1]])
        size = len(obs)
        return obs, size


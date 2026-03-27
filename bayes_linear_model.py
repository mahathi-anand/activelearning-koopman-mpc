import numpy as np


class BayesianLinearRegressionDynamics:
    """
    Bayesian linear regression over theta = vec([A B]) with:
        x_{k+1} = [A B] [x_k; u_k] + w_k
    """

    def __init__(self, task):
        self.n_x = task.n_x
        self.n_u = task.n_u
        self.p = self.n_x + self.n_u
        self.n_theta = self.n_x * self.p

        self.Sigma_w = task.process_noise_cov.copy()
        self.Sigma_w_inv = np.linalg.inv(self.Sigma_w)

        self.mu = self.pack_theta(task.A_prior_mean, task.B_prior_mean)
        theta_std = np.hstack([task.A_prior_std, task.B_prior_std]).reshape(-1, order="F")
        self.P = np.diag(np.maximum(theta_std**2, 1e-8))

    def pack_theta(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        M = np.hstack([A, B])
        return M.reshape(-1, order="F")

    def unpack_theta(self, theta: np.ndarray):
        M = theta.reshape((self.n_x, self.p), order="F")
        A = M[:, :self.n_x]
        B = M[:, self.n_x:]
        return A, B

    def H_of_psi(self, psi: np.ndarray) -> np.ndarray:
        return np.kron(psi.reshape(1, -1), np.eye(self.n_x))

    def posterior_update(self, x_k: np.ndarray, u_k: np.ndarray, x_next: np.ndarray):
        psi = np.hstack([x_k, u_k])
        H = self.H_of_psi(psi)

        S_mat = self.Sigma_w + H @ self.P @ H.T
        K_gain = self.P @ H.T @ np.linalg.inv(S_mat)

        self.mu = self.mu + K_gain @ (x_next - H @ self.mu)
        self.P = self.P - K_gain @ H @ self.P
        self.P = 0.5 * (self.P + self.P.T)

        evals, evecs = np.linalg.eigh(self.P)
        evals = np.maximum(evals, 1e-12)
        self.P = (evecs * evals) @ evecs.T
        self.P = 0.5 * (self.P + self.P.T)

    def sample_dynamics(self, rng: np.random.Generator):
        theta_s = rng.multivariate_normal(self.mu, self.P)
        return self.unpack_theta(theta_s)

    def posterior_weight_matrix(self) -> np.ndarray:
        P2 = self.P @ self.P
        W = np.zeros((self.p, self.p))
        for i in range(self.n_x):
            for j in range(self.n_x):
                block_ij = P2[i * self.p:(i + 1) * self.p, j * self.p:(j + 1) * self.p]
                W += self.Sigma_w_inv[i, j] * block_ij
        W = 0.5 * (W + W.T)

        evals, evecs = np.linalg.eigh(W)
        evals = np.maximum(evals, 0.0)
        W = (evecs * evals) @ evecs.T
        return 0.5 * (W + W.T)

import numpy as np


def polygon_to_halfspaces(vertices: np.ndarray):
    """Convert a CCW polygon into halfspaces H p <= h."""
    m = vertices.shape[0]
    H = []
    h = []
    for i in range(m):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % m]
        edge = v2 - v1
        normal = np.array([edge[1], -edge[0]], dtype=float)
        rhs = normal @ v1
        H.append(normal)
        h.append(rhs)
    return np.array(H), np.array(h)


class Figure8GravityTask:
    def __init__(self):
        self.dt = 0.25

        self.A_true = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.B_true = np.array([
            [0.5 * self.dt**2, 0.0],
            [0.0, 0.5 * self.dt**2],
            [self.dt, 0.0],
            [0.0, self.dt],
        ])
        self.n_x = self.A_true.shape[0]
        self.n_u = self.B_true.shape[1]
        self.process_noise_cov = np.diag([1e-4, 1e-4, 1e-4, 1e-4])

        # Priors used by the Bayesian learner.
        self.A_bayes_mean = self.A_true.copy() + 0.1
        self.B_bayes_mean = self.B_true.copy() + 0.1
        self.A_bayes_std = 0.1 * np.abs(self.A_true) + 0.1
        self.B_bayes_std = 0.1 * np.abs(self.B_true) + 0.1
        self.A_eval = self.A_true.copy()
        self.B_eval = self.B_true.copy()

        # Figure-8 reference
        self.ref_center = np.array([0.0, 0.0])
        self.ref_amp_x = 4.0
        self.ref_amp_y = 2.0
        self.ref_omega = 0.20
        self.ref_phase = np.pi

        self.x_init = self.reference_state(0)

        # Baked fixed polygon (CCW): exact scaled+centered version of
        # the original shape, with no extra deformation.
        self.safe_vertices = np.array([
            [-4.5283, -3.0000],
            [ 1.5217, -3.0000],
            [ 4.4917, -0.6855],
            [ 4.4917,  3.0000],
            [-1.4483,  3.0000],
            [-4.5283,  0.6855],
        ])

        H_p, h_p = polygon_to_halfspaces(self.safe_vertices)
        self.H_x_full = np.hstack([H_p, np.zeros((H_p.shape[0], 2))])
        self.h_p = h_p

    def reference_state(self, step_idx: int) -> np.ndarray:
        t = step_idx * self.dt
        px = self.ref_center[0] + self.ref_amp_x * np.sin(self.ref_omega * t + self.ref_phase)
        py = self.ref_center[1] + self.ref_amp_y * np.sin(2.0 * self.ref_omega * t + 2.0 * self.ref_phase)
        vx = self.ref_amp_x * self.ref_omega * np.cos(self.ref_omega * t + self.ref_phase)
        vy = 2.0 * self.ref_amp_y * self.ref_omega * np.cos(2.0 * self.ref_omega * t + 2.0 * self.ref_phase)
        return np.array([px, py, vx, vy])

    def true_step(self, x: np.ndarray, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        w = rng.multivariate_normal(np.zeros(self.n_x), self.process_noise_cov)
        return self.A_true @ x + self.B_true @ u + w


class KoopmanVanDerPolTask:
    """
    Van der Pol task with lifted (Koopman-style) observables.

    Physical state:
        x_phys = [x0, x1]

    Lifted state used by MPC/learning:
        z = [x0, x1, x0^2, x1^2, x0^2 * x1]
    """

    def __init__(self, mu: float = 1.0, dt: float = 0.01):
        self.system_name = "VanDerPolKoopman"
        self.mu = float(mu)
        self.dt = float(dt)

        self.n_x_phys = 2
        self.n_u = 1
        self.n_x = 5

        # Two fitted lifted models:
        # - prior fit: smaller dataset for Bayesian initialization
        # - eval fit: larger dataset for diagnostics
        self.A_lift_prior_fit, self.B_lift_prior_fit = self._fit_lifted_linear_model(
            num_x0=10, num_x1=10, num_u=10
        )
        self.A_lift_eval_fit, self.B_lift_eval_fit = self._fit_lifted_linear_model(
            num_x0=10, num_x1=10, num_u=10
        )

        # Process noise is modeled in lifted coordinates.
        self.process_noise_cov = np.diag([1e-5, 1e-5, 2e-5, 2e-5, 2e-5])

        # Priors for Bayesian learner.
        self.A_bayes_mean = self.A_lift_prior_fit.copy()
        self.B_bayes_mean = self.B_lift_prior_fit.copy()
        self.A_bayes_std = 0.01 * np.abs(self.A_lift_prior_fit) + 0.
        self.B_bayes_std = 0.01 * np.abs(self.B_lift_prior_fit) + 0.
        self.A_eval = self.A_lift_eval_fit.copy()
        self.B_eval = self.B_lift_eval_fit.copy()

        # Figure-8 reference in physical coordinates, then lifted.
        self.ref_center = np.array([0.0, 0.0])
        self.ref_amp_x = 1.8
        self.ref_amp_y = 1.2
        self.ref_omega = 0.9
        self.ref_phase = np.pi

        # Constraints on physical coordinates: x0,x1 in [-1, 1].
        self.safe_vertices = np.array([
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ])
        H_p, h_p = polygon_to_halfspaces(self.safe_vertices)
        self.H_x_full = np.hstack([H_p, np.zeros((H_p.shape[0], self.n_x - 2))])
        self.h_p = h_p

        self.x_init = self.reference_state(0)

    def dynamics(self, x_phys: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Nonlinear Van der Pol dynamics in physical 2D coordinates."""
        u0 = float(u[0])
        x0 = x_phys[0] + self.dt * x_phys[1]
        x1 = x_phys[1] + self.dt * (
            self.mu * (1.0 - x_phys[0] ** 2) * x_phys[1] - x_phys[0] + u0
        )
        return np.array([x0, x1], dtype=float)

    def observables(self, x_phys: np.ndarray) -> np.ndarray:
        x0, x1 = float(x_phys[0]), float(x_phys[1])
        return np.array([x0, x1, x0 * x0, x1 * x1, x0 * x0 * x1], dtype=float)

    def unlift_state(self, z: np.ndarray) -> np.ndarray:
        """Recover physical coordinates from lifted state."""
        return np.asarray(z[:2], dtype=float)

    def reference_physical_state(self, step_idx: int) -> np.ndarray:
        t = step_idx * self.dt
        x0 = self.ref_center[0] + self.ref_amp_x * np.sin(self.ref_omega * t + self.ref_phase)
        x1 = self.ref_center[1] + self.ref_amp_y * np.sin(
            2.0 * self.ref_omega * t + 2.0 * self.ref_phase
        )
        return np.array([x0, x1], dtype=float) * 0.

    def reference_state(self, step_idx: int) -> np.ndarray:
        return self.observables(self.reference_physical_state(step_idx))

    def true_step(self, z: np.ndarray, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Propagate the nonlinear physical dynamics, then re-lift.
        This keeps lifted dynamics transparent to the rest of the stack.
        """
        x_phys = self.unlift_state(z)
        x_next = self.dynamics(x_phys, u)
        z_next = self.observables(x_next)
        w = rng.multivariate_normal(np.zeros(self.n_x), self.process_noise_cov)
        return z_next + w

    def _fit_lifted_linear_model(self, num_x0: int = 15, num_x1: int = 15, num_u: int = 9):
        """
        Fit A,B in z_{k+1} = A z_k + B u_k by ridge least squares over a grid.
        """
        x0_grid = np.linspace(-2.2, 2.2, num_x0)
        x1_grid = np.linspace(-1.8, 1.8, num_x1)
        u_grid = np.linspace(-1.0, 1.0, num_u)

        psi_cols = []
        y_cols = []
        for x0 in x0_grid:
            for x1 in x1_grid:
                x_phys = np.array([x0, x1], dtype=float)
                for u0 in u_grid:
                    u = np.array([u0], dtype=float)
                    z = self.observables(x_phys)
                    x_next = self.dynamics(x_phys, u)
                    z_next = self.observables(x_next)
                    psi_cols.append(np.hstack([z, u]))
                    y_cols.append(z_next)

        Psi = np.column_stack(psi_cols)  # (n_x+n_u, n_samples)
        Y = np.column_stack(y_cols)      # (n_x, n_samples)

        ridge = 1e-6
        gram = Psi @ Psi.T + ridge * np.eye(self.n_x + self.n_u)
        M = (Y @ Psi.T) @ np.linalg.inv(gram)
        A = M[:, :self.n_x]
        B = M[:, self.n_x:]
        return A, B

import numpy as np

# Utility function
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

class ReferenceTask:
    def __init__(self, 
                 system, 
                 noise_cov = 1e-2, 
                 prior_mean = 0.1, prior_std = 0.1, 
                 reference_fn = None):
        self.A_true = system.A.copy()
        self.B_true = system.B.copy()

        if self.A_true is None or self.B_true is None:
            raise ValueError("system must provide A and B matrices")

        self.n_x = system.n_x
        self.n_u = system.n_u
        self.dt = system.dt if hasattr(system, "dt") else 0.01

        if noise_cov is None:
            self.process_noise_cov = np.diag([1e-2] * self.n_x)
        else:
            self.process_noise_cov = np.diag([noise_cov] * self.n_x)

        #self.process_noise_cov = noise_cov

        #Priors used by Bayesian
        self.A_prior_mean = self.A_true.copy() + prior_mean
        self.B_prior_mean = self.B_true.copy() + prior_mean
        self.A_prior_std = prior_std * np.abs(self.A_true) + prior_std
        self.B_prior_std = prior_std * np.abs(self.B_true) + prior_std

        #Reference trajectory function
        if reference_fn is None:
            self.reference_state = lambda step_idx: np.zeros(self.n_x)
        else:
            if not callable(reference_fn):
                raise ValueError("reference_fn must be a callable function")
            
            def reference_wrapper(step_idx):
                try:
                    return reference_fn(self,step_idx)
                except TypeError:
                    return reference_fn(step_idx)

            self.reference_state = reference_wrapper
                    
        self.x_init = np.random.uniform(self.plot_bounds[0], self.plot_bounds[1], size=self.n_x) if hasattr(self, "plot_bounds") else np.zeros(self.n_x)
   
    #True dynamics under disturbance
    def true_step(self, x: np.ndarray, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        w = rng.multivariate_normal(np.zeros(self.n_x), self.process_noise_cov)
        return self.A_true @ x + self.B_true @ u + w

    #Define reference trajectories here. For example, a figure-8 reference:
    def fig_eight(self, step_idx: int) -> np.ndarray:
        #parameters for the figure-8 trajectory
        ref_center = np.array([0.0, 0.0])
        ref_amp_x = 4.0
        ref_amp_y = 2.0
        ref_omega = 0.20
        ref_phase = np.pi
        
        t = step_idx * self.dt
        px = ref_center[0] + ref_amp_x * np.sin(ref_omega * t + ref_phase)
        py = ref_center[1] + ref_amp_y * np.sin(2.0 * ref_omega * t + 2.0 * ref_phase)
        vx = ref_amp_x * ref_omega * np.cos(ref_omega * t + ref_phase)
        vy = 2.0 * ref_amp_y * ref_omega * np.cos(2.0 * ref_omega * t + 2.0 * ref_phase)
        return np.array([px, py, vx, vy])

    #Any supporting functions for reference trajectory generation can be added here as well. 


class Figure8GravityTask:
    def __init__(self):
        self.dt = 0.25

        self.A_nom = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.B_nom = np.array([
            [0.5 * self.dt**2, 0.0],
            [0.0, 0.5 * self.dt**2],
            [self.dt, 0.0],
            [0.0, self.dt],
        ])
        self.n_x = self.A_nom.shape[0]
        self.n_u = self.B_nom.shape[1]
        self.A_true = self.A_nom.copy()
        self.B_true = self.B_nom.copy()
        self.process_noise_cov = np.diag([1e-4, 1e-4, 1e-4, 1e-4])

        # Priors used by the Bayesian learner.
        self.A_prior_mean = self.A_nom.copy() + 0.1
        self.B_prior_mean = self.B_nom.copy() + 0.1
        self.A_prior_std = 0.1 * np.abs(self.A_nom) + 0.1
        self.B_prior_std = 0.1 * np.abs(self.B_nom) + 0.1

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




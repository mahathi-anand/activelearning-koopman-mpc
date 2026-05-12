from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

#Utility function to compute safety set
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

@dataclass
class SimpleMPCConfig:
    N: int = 15
    S: int = 10
    num_applied_steps: int = 200
    actions_per_mpc_solve: int = 3
    num_scenarios_to_plot: int = 8
    plotting_dimension: int = 2 #1 or 2 depending on position dimension
    plot_bounds: tuple = (-5, 5, -5, 5)

    u_max: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    v_max: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0]))

    Q: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 0.5, 0.5]))
    Qf: np.ndarray = field(default_factory=lambda: np.diag([20.0, 20.0, 1.0, 1.0]))
    R: np.ndarray = field(default_factory=lambda: 0.08 * np.eye(2))

    safety_vertices: np.ndarray = field(default_factory=lambda: np.array([
            [-4.5283, -3.0000],
            [ 1.5217, -3.0000],
            [ 4.4917, -0.6855],
            [ 4.4917,  3.0000],
            [-1.4483,  3.0000],
            [-4.5283,  0.6855]]))
    
    H_p: np.ndarray = field(default_factory = lambda: None)
    h_p: np.ndarray = field(default_factory = lambda: None)
    H_x_full : np.ndarray = field(default_factory = lambda: None)

    cost_weight: float = 1.0
    info_gain_weight: float = 100.0
    slack_weight: float = 10000.0
    velocity_slack_weight: float = 10000.0

    solver: str = "CLARABEL"
    solver_opts: dict = field(default_factory=lambda: {"verbose": False})

    def __post_init__(self):
        if self.H_p is None or self.h_p is None:
            self.H_p, self.h_p = polygon_to_halfspaces(self.safety_vertices)

        if self.H_x_full is None:
            self.H_x_full = np.hstack([self.H_p, np.zeros((self.H_p.shape[0], 2))])

@dataclass
class KoopmanVanDerPolConfig(SimpleMPCConfig):
   system: Optional[Any] = field(default=None, repr=False)
   use_info_gain: bool = field(default=True, repr=False) #Activate or deactivate information gain in the MPC objective
   Q: np.ndarray = field(default_factory = lambda: None)
   Qf: np.ndarray = field(default_factory = lambda: None)
   R: np.ndarray = field(default_factory = lambda: None)
   safety_vertices: np.ndarray = field(default_factory=lambda: None)
   plotting_dimension: int = 1 #1 or 2 depending on position dimension
   plot_bounds: tuple = (-5,5)
   #For 1D case, safety set is defined by these bounds. For 2D case, safety set is defined by vertices or halfspace constraints.
   safety_bounds: tuple = (-3, 3) 

   #Input constraints if needed
   u_max: np.ndarray = field(default_factory=lambda: None)

   #Setting up optimization matrices according to observable size 
   def __post_init__(self):
        
        if self.system is not None:
            n_x = getattr(self.system, 'n_x') 
            n_u = getattr(self.system, 'n_u')
        else:
            n_x = 5
            n_u = 1

        #Optimization matrices    
        if self.Q is None:
           D1 = np.eye(2)
           D2 = np.zeros((2, n_x - 2))
           D3 = np.zeros((n_x - 2, 2))
           D4 = np.zeros((n_x - 2, n_x - 2))
           self.Q = np.block([[D1, D2], [D3, D4]])
        
        if self.Qf is None:
            self.Qf = np.zeros((n_x, n_x))

        if self.R is None:
            self.R = 0.1 * np.eye(n_u)

        if self.u_max is None:
            self.u_max = 2.0 * np.ones((n_u, 1))

        if self.H_p is None or self.h_p is None:
            #Add Safety Constraints. For Van der Pol Oscillator, -3 <= x_1 <= 3
            self.H_p = np.array([[1.0, 0.0], [-1.0, 0.0]])
            self.h_p = np.array([self.safety_bounds[1], -self.safety_bounds[0]])
            #Polyhedral safety constraints if needed
             #   self.H_p, self.h_p = polygon_to_halfspaces(self.safety_vertices)

        #Make sure H_x is consistent with the observable state dimension
        if self.H_x_full is None:
            self.H_x_full = np.hstack([self.H_p, np.zeros((self.H_p.shape[0], n_x - self.H_p.shape[1]))])
    

   
    

        
        
    


    

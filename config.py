from dataclasses import dataclass, field
import numpy as np


@dataclass
class MPCConfig:
    N: int = 15
    S: int = 10
    num_applied_steps: int = 200
    actions_per_mpc_solve: int = 3
    num_scenarios_to_plot: int = 8

    u_max: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    v_max: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0]))

    Q: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 0.5, 0.5]))
    Qf: np.ndarray = field(default_factory=lambda: np.diag([20.0, 20.0, 1.0, 1.0]))
    R: np.ndarray = field(default_factory=lambda: 0.08 * np.eye(2))

    cost_weight: float = 1.0
    info_gain_weight: float = 100.0
    slack_weight: float = 10000.0
    velocity_slack_weight: float = 10000.0

    solver: str = "CLARABEL"
    solver_opts: dict = field(default_factory=lambda: {"verbose": False})

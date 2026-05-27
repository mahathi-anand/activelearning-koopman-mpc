from dataclasses import dataclass, field
import numpy as np


@dataclass
class Figure8MPCConfig:
    N: int = 15
    S: int = 10
    num_applied_steps: int = 200
    actions_per_mpc_solve: int = 3
    num_scenarios_to_plot: int = 8

    u_min: np.ndarray = field(default_factory=lambda: np.array([-1.0, -1.0]))
    u_max: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    s_min: np.ndarray = field(
        default_factory=lambda: np.array([-np.inf, -np.inf, -2.0, -2.0])
    )
    s_max: np.ndarray = field(
        default_factory=lambda: np.array([np.inf, np.inf, 2.0, 2.0])
    )

    Q: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 0.5, 0.5]))
    Qf: np.ndarray = field(default_factory=lambda: np.diag([20.0, 20.0, 1.0, 1.0]))
    R: np.ndarray = field(default_factory=lambda: 0.08 * np.eye(2))

    cost_weight: float = 1.0
    info_gain_weight: float = 100.0
    slack_weight: float = 10000.0

    solver: str = "CLARABEL"
    solver_opts: dict = field(default_factory=lambda: {"verbose": False})


@dataclass
class KoopmanVanDerPolMPCConfig:
    N: int = 15
    S: int = 10
    num_applied_steps: int = 2_000
    actions_per_mpc_solve: int = 3
    num_scenarios_to_plot: int = 8

    # Appendix: per-dimension bounds for z = [x0, x1, x0^2, x1^2, x0^2*x1].
    # Use +/- np.inf for unconstrained dimensions.
    u_min: np.ndarray = field(default_factory=lambda: np.array([-2.0]))  # [u]
    u_max: np.ndarray = field(default_factory=lambda: np.array([2.0]))   # [u]
    s_min: np.ndarray = field(
        default_factory=lambda: np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    )
    s_max: np.ndarray = field(
        default_factory=lambda: np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
    )

    # Lifted state z = [x0, x1, x0^2, x1^2, x0^2*x1]
    # Track physical coordinates strongly; keep lifted terms lightly regularized.
    Q: np.ndarray = field(default_factory=lambda: np.diag([12.0, 12.0, 0.25, 0.25, 0.25]))
    Qf: np.ndarray = field(default_factory=lambda: np.diag([24.0, 24.0, 0.5, 0.5, 0.5]))
    R: np.ndarray = field(default_factory=lambda: np.array([[0.04]]))

    cost_weight: float = 1.0
    info_gain_weight: float = 100.0
    slack_weight: float = 10000.0

    solver: str = "CLARABEL"
    solver_opts: dict = field(default_factory=lambda: {"verbose": False})

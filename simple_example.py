import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ============================================================
# Scenario MPC for a 2D double integrator with
# fixed convex state constraints and soft constraint violation
# ============================================================

# State: x = [px, py, vx, vy]
# Input: u = [ax, ay]

# ----------------------------
# Dynamics
# ----------------------------
dt = 0.25

A = np.array([
    [1.0, 0.0, dt,  0.0],
    [0.0, 1.0, 0.0, dt ],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

B = np.array([
    [0.5 * dt**2, 0.0],
    [0.0, 0.5 * dt**2],
    [dt, 0.0],
    [0.0, dt],
])

A_nom = A.copy()
B_nom = B.copy()

# ----------------------------
# MPC parameters
# ----------------------------
N = 20          # horizon
S = 8           # number of scenarios
num_applied_steps = 50  # number of MPC steps applied to the true system
actions_per_mpc_solve = 1  # apply this many planned controls before re-solving MPC
num_scenarios_to_plot = 8  # closed-loop scenario trajectories shown in XY plot

u_max = np.array([1.0, 1.0])
v_max = np.array([2.0, 2.0])

Q = np.diag([10.0, 10.0, 0.5, 0.5])
Qf = np.diag([20.0, 20.0, 1.0, 1.0])
R = 0.08 * np.eye(2)

cost_weight = 1.0
info_gain_weight = 10.0
# Weighted excitation dimensions for phi = [px, py, vx, vy, ax, ay].
W_phi = np.diag([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
info_gain_eps = 1e-3

slack_weight = 10000.0
velocity_slack_weight = 10000.0
# solver = cp.OSQP
# solver_opts = {
#     "warm_start": True,
#     "verbose": False,
#     "eps_abs": 1e-4,
#     "eps_rel": 1e-4,
#     "max_iter": 20000,
# }

solver = cp.CLARABEL
solver_opts = {
    "verbose": False,
}

# Scenario dynamics uncertainty: A_s ~ N(A_nom, A_std), B_s ~ N(B_nom, B_std)
# Keep perturbations small (especially on zero entries) to avoid frequent infeasibility.
A_std = 0.01 * np.abs(A_nom)
B_std = 0.001 * np.abs(B_nom)

# ----------------------------
# Initial state and goal
# ----------------------------
# Nonzero initial velocity points toward the lower boundary.
x_init = np.array([-0.0, -2.25, 0.0, -1.5])
goal = np.array([3.5, 1.15])

# ----------------------------
# Fixed convex safe region in (px, py)
# Hexagon-like polygon
# ----------------------------
safe_vertices = np.array([
    [-4.0, -3.5],
    [ 1.5, -3.5],
    [ 4.2, -0.8],
    [ 4.2,  3.5],
    [-1.2,  3.5],
    [-4.0,  0.8],
])

# Convert polygon to halfspace form H_p p <= h_p
# Assumes vertices are ordered counterclockwise.
def polygon_to_halfspaces(vertices):
    m = vertices.shape[0]
    H = []
    h = []
    for i in range(m):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % m]
        edge = v2 - v1

        # outward normal for CCW polygon
        normal = np.array([edge[1], -edge[0]], dtype=float)
        rhs = normal @ v1

        H.append(normal)
        h.append(rhs)

    return np.array(H), np.array(h)

H_p, h_p = polygon_to_halfspaces(safe_vertices)
m_poly = H_p.shape[0]

# Embed position-only constraints into full state
# H_x_full * x <= h_p + slack
H_x_full = np.hstack([H_p, np.zeros((m_poly, 2))])

# ----------------------------
# CVXPY variables and parameters
# ----------------------------
U = cp.Variable((2, N))
X = [cp.Variable((4, N + 1)) for _ in range(S)]
slack = [cp.Variable((m_poly, N + 1), nonneg=True) for _ in range(S)]
vel_slack = [cp.Variable((2, N + 1), nonneg=True) for _ in range(S)]

x0_param = cp.Parameter(4)
goal_param = cp.Parameter(2)
A_params = [cp.Parameter((4, 4)) for _ in range(S)]
B_params = [cp.Parameter((4, 2)) for _ in range(S)]
phi_grad_params = [cp.Parameter((6, N)) for _ in range(S)]
phi_recip_coeff_params = [cp.Parameter(N, nonneg=True) for _ in range(S)]

goal_state = cp.hstack([goal_param, np.zeros(2)])

constraints = []
tracking_cost = 0.0
info_recip_surrogate = 0.0

# Shared input constraints
for k in range(N):
    constraints += [
        U[:, k] <= u_max,
        U[:, k] >= -u_max,
    ]
    tracking_cost += cp.quad_form(U[:, k], R)

# Scenario constraints and cost
for s in range(S):
    constraints += [X[s][:, 0] == x0_param]

    for k in range(N):
        # scenario-specific uncertain dynamics
        constraints += [
            X[s][:, k + 1] == A_params[s] @ X[s][:, k] + B_params[s] @ U[:, k]
        ]

        # velocity constraints
        constraints += [
            X[s][2:, k] <= v_max + vel_slack[s][:, k],
            X[s][2:, k] >= -v_max - vel_slack[s][:, k],
        ]

        # soft fixed convex safe-set constraints
        constraints += [
            H_x_full @ X[s][:, k] <= h_p + slack[s][:, k]
        ]
        tracking_cost += (1.0 / S) * cp.quad_form(X[s][:, k] - goal_state, Q)
        tracking_cost += slack_weight * cp.sum(slack[s][:, k])
        tracking_cost += velocity_slack_weight * cp.sum(vel_slack[s][:, k])
        # Sequential linearization of + w / (phi^T W_phi phi + eps):
        # f(g)=w/g, g>0; f(g) ~= const - (w / g_ref^2) * g_lin.
        info_recip_surrogate += (1.0 / S) * phi_recip_coeff_params[s][k] * (
            cp.sum(cp.multiply(phi_grad_params[s][:4, k], X[s][:, k]))
            + cp.sum(cp.multiply(phi_grad_params[s][4:, k], U[:, k]))
        )

    # terminal constraints/cost
    constraints += [
        X[s][2:, N] <= v_max + vel_slack[s][:, N],
        X[s][2:, N] >= -v_max - vel_slack[s][:, N],
        H_x_full @ X[s][:, N] <= h_p + slack[s][:, N],
    ]

    tracking_cost += (1.0 / S) * cp.quad_form(X[s][:, N] - goal_state, Qf)
    tracking_cost += slack_weight * cp.sum(slack[s][:, N])
    tracking_cost += velocity_slack_weight * cp.sum(vel_slack[s][:, N])
total_objective = cost_weight * tracking_cost - info_recip_surrogate
problem = cp.Problem(cp.Minimize(total_objective), constraints)

# ----------------------------
# Closed-loop simulation
# ----------------------------
def sample_dynamics(rng, scale=1.0):
    A_s = rng.normal(loc=A_nom, scale=scale * A_std)
    B_s = rng.normal(loc=B_nom, scale=scale * B_std)
    return A_s, B_s


def run_closed_loop(x0, rng):
    x_true = x0.copy()
    xs = [x_true.copy()]
    us = []

    # Sample scenario systems once for this rollout and keep them fixed.
    scenario_As = []
    scenario_Bs = []
    for s in range(S):
        A_s, B_s = sample_dynamics(rng, scale=1.0)
        scenario_As.append(A_s)
        scenario_Bs.append(B_s)

    scenario_branches = []
    x_info_refs = [np.tile(x_true.reshape(4, 1), (1, N + 1)) for _ in range(S)]
    u_info_ref = np.zeros((2, N))

    applied_steps = 0
    while applied_steps < num_applied_steps:
        x0_param.value = x_true
        goal_param.value = goal

        for s in range(S):
            A_params[s].value = scenario_As[s]
            B_params[s].value = scenario_Bs[s]
            phi_grad = np.zeros((6, N))
            phi_recip_coeff = np.zeros(N)
            for k in range(N):
                phi_ref = np.hstack([x_info_refs[s][:, k], u_info_ref[:, k]])
                wphi_ref = W_phi @ phi_ref
                g_ref = float(phi_ref @ wphi_ref) + info_gain_eps
                phi_grad[:, k] = 2.0 * wphi_ref
                phi_recip_coeff[k] = info_gain_weight / (g_ref * g_ref)
            phi_grad_params[s].value = phi_grad
            phi_recip_coeff_params[s].value = phi_recip_coeff

        problem.solve(solver=solver, **solver_opts)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC solve failed at step {applied_steps}: {problem.status}")

        for s in range(S):
            x_info_refs[s] = np.asarray(X[s].value)
        u_info_ref = np.asarray(U.value)

        u_plan = np.asarray(U.value)
        remaining_steps = num_applied_steps - applied_steps
        controls_to_apply = min(actions_per_mpc_solve, N, remaining_steps)

        # Store full MPC-horizon scenario rollout shown from this node.
        branch = np.stack(
            [np.asarray(X[s].value.T) for s in range(S)],
            axis=0,
        )  # (S, N+1, 4)
        scenario_branches.append(branch)

        for k in range(controls_to_apply):
            u = u_plan[:, k].reshape(2)
            us.append(u.copy())

            # True plant uses nominal dynamics.
            x_true = A_nom @ x_true + B_nom @ u
            xs.append(x_true.copy())

            applied_steps += 1

    return np.asarray(xs), np.asarray(us), scenario_branches


rng = np.random.default_rng(4)
xs_true, us_true, scenario_branches = run_closed_loop(x_init, rng)

# ----------------------------
# Plot
# ----------------------------
fig, (ax_xy, ax_u) = plt.subplots(1, 2, figsize=(13, 5))

# safe region
poly = Polygon(safe_vertices, closed=True, fill=False, linewidth=2)
ax_xy.add_patch(poly)

# scenario prediction tree (thinner dashed branches from each true state node)
for branch_idx, branch in enumerate(scenario_branches):
    num_scenarios_plotted = min(num_scenarios_to_plot, branch.shape[0])
    for scenario_idx in range(num_scenarios_plotted):
        label = "scenario branches" if (branch_idx == 0 and scenario_idx == 0) else None
        ax_xy.plot(
            branch[scenario_idx, :, 0],
            branch[scenario_idx, :, 1],
            "--",
            linewidth=1.0,
            alpha=0.35,
            color="tab:orange",
            label=label,
        )

# true closed-loop trajectory (bold)
ax_xy.plot(xs_true[:, 0], xs_true[:, 1], color="tab:blue", linewidth=3.0, label="true state")

ax_xy.scatter(x_init[0], x_init[1], marker="s", s=80, label="start")
ax_xy.scatter(goal[0], goal[1], marker="*", s=120, label="goal")

ax_xy.set_title("2D scenario MPC with fixed convex safe set")
ax_xy.set_xlabel("x")
ax_xy.set_ylabel("y")

# Keep a meaningful viewport even if some sampled scenario rollouts diverge.
x_min = min(np.min(safe_vertices[:, 0]), np.min(xs_true[:, 0])) - 0.5
x_max = max(np.max(safe_vertices[:, 0]), np.max(xs_true[:, 0])) + 0.5
y_min = min(np.min(safe_vertices[:, 1]), np.min(xs_true[:, 1])) - 0.5
y_max = max(np.max(safe_vertices[:, 1]), np.max(xs_true[:, 1])) + 0.5
ax_xy.set_aspect("equal", adjustable="box")
ax_xy.set_xlim(x_min, x_max)
ax_xy.set_ylim(y_min, y_max)
ax_xy.autoscale(False)
ax_xy.grid(True)
ax_xy.legend()

# input plot (true system rollout)
if len(us_true) > 0:
    t_u = np.arange(len(us_true))
    ax_u.step(t_u, us_true[:, 0], where="post", label="ax")
    ax_u.step(t_u, us_true[:, 1], where="post", label="ay")
    ax_u.axhline(u_max[0], linestyle="--", linewidth=1.0)
    ax_u.axhline(-u_max[0], linestyle="--", linewidth=1.0)
    ax_u.set_title("Applied control inputs")
    ax_u.set_xlabel("time step")
    ax_u.set_ylabel("u")
    ax_u.grid(True)
    ax_u.legend()

plt.tight_layout()
plt.show()

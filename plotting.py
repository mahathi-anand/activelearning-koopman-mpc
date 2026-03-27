import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def plot_run(task, cfg, results):
    xs_true = results["xs_true"]
    x_refs = results["x_refs"]
    us_true = results["us_true"]
    scenario_branches = results["scenario_branches"]
    mu_history = results["mu_history"]
    trace_history = results["trace_history"]

    fig, (ax_xy, ax_u, ax_x, ax_err) = plt.subplots(1, 4, figsize=(24, 5))

    poly = Polygon(task.safe_vertices, closed=True, fill=False, linewidth=2)
    ax_xy.add_patch(poly)

    for branch_idx, branch in enumerate(scenario_branches):
        num_scenarios_plotted = min(cfg.num_scenarios_to_plot, branch.shape[0])
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

    ax_xy.plot(xs_true[:, 0], xs_true[:, 1], color="tab:blue", linewidth=3.0, label="true state")
    ax_xy.scatter(task.x_init[0], task.x_init[1], marker="s", s=80, label="start")

    ref_plot = np.array([task.reference_state(k) for k in range(cfg.num_applied_steps + 1)])
    ax_xy.plot(
        ref_plot[:, 0], ref_plot[:, 1],
        color="tab:green", linewidth=2.0, alpha=0.9, label="figure-8 reference"
    )

    ax_xy.set_title("Posterior-aware scenario MPC (modular)")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")

    x_min = min(np.min(task.safe_vertices[:, 0]), np.min(xs_true[:, 0])) - 0.5
    x_max = max(np.max(task.safe_vertices[:, 0]), np.max(xs_true[:, 0])) + 0.5
    y_min = min(np.min(task.safe_vertices[:, 1]), np.min(xs_true[:, 1])) - 0.5
    y_max = max(np.max(task.safe_vertices[:, 1]), np.max(xs_true[:, 1])) + 0.5
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlim(x_min, x_max)
    ax_xy.set_ylim(y_min, y_max)
    ax_xy.grid(True)
    ax_xy.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0.0)

    if len(us_true) > 0:
        t_u = np.arange(len(us_true))
        ax_u.step(t_u, us_true[:, 0], where="post", label="ax")
        ax_u.step(t_u, us_true[:, 1], where="post", label="ay")
        ax_u.axhline(cfg.u_max[0], linestyle="--", linewidth=1.0)
        ax_u.axhline(-cfg.u_max[0], linestyle="--", linewidth=1.0)
        ax_u.set_title("Applied control inputs")
        ax_u.set_xlabel("time step")
        ax_u.set_ylabel("u")
        ax_u.grid(True)
        ax_u.legend()

    if len(xs_true) > 0 and len(x_refs) > 0:
        t_x = np.arange(len(xs_true))
        state_labels = ["px", "py"]
        for i, label in enumerate(state_labels):
            ax_x.plot(t_x, xs_true[:, i], linewidth=1.6, label=f"{label}")
            ax_x.plot(t_x, x_refs[:, i], "--", linewidth=1.2, alpha=0.8, label=f"{label}_ref")
        ax_x.set_title("Position vs References")
        ax_x.set_xlabel("time step")
        ax_x.set_ylabel("position")
        ax_x.grid(True)
        ax_x.legend(ncol=2)

    if len(mu_history) > 0:
        theta_true = np.hstack([task.A_true, task.B_true]).reshape(-1, order="F")
        theta_err = mu_history - theta_true.reshape(1, -1)
        t_e = np.arange(theta_err.shape[0])

        n_a = task.n_x * task.n_x
        for i in range(n_a):
            ax_err.plot(t_e, theta_err[:, i], linewidth=0.9, alpha=0.7, color="tab:blue")
        for i in range(n_a, theta_err.shape[1]):
            ax_err.plot(t_e, theta_err[:, i], linewidth=0.9, alpha=0.7, color="tab:orange")

        ax_err.set_title("Parameter Estimation Error")
        ax_err.set_xlabel("time step")
        ax_err.set_ylabel("mu - true")
        ax_err.grid(True)
        ax_err.plot([], [], color="tab:blue", label="A entries")
        ax_err.plot([], [], color="tab:orange", label="B entries")
        ax_err.legend()

        if len(trace_history) > 0:
            ax_trace = ax_err.twinx()
            ax_trace.plot(t_e, trace_history, color="black", linewidth=2.0, alpha=0.9, label="trace(P)")
            ax_trace.set_ylabel("trace(P)")
            ax_trace.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

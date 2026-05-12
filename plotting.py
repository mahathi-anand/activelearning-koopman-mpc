import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# Utility functions 
def halfspace_polygon(H, h, config):
    """Clip a bounding box by halfspaces H x <= h; return 2D polygon (Mx2)."""
    (xmin, xmax, ymin, ymax) = config.plot_bounds        
    poly = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]], dtype=float)
    for Hi, hi in zip(H, h):
        if poly.size == 0:
            break
        out, n = [], len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i+1) % n]
            in_a = Hi.dot(a) <= hi + 1e-12
            in_b = Hi.dot(b) <= hi + 1e-12
            if in_a and in_b:
                out.append(b)
            elif in_a:
                d = Hi.dot(b - a)
                if abs(d) > 1e-12: out.append(a + (hi - Hi.dot(a)) / d * (b - a))
            elif in_b:
                d = Hi.dot(b - a)
                if abs(d) > 1e-12: out.append(a + (hi - Hi.dot(a)) / d * (b - a))
                out.append(b)
        poly = np.vstack(out) if out else np.zeros((0, 2))
    return poly


def plot_traj(task, config, results):
    xs_true = results["xs_true"]
    x_refs = results["x_refs"]
    us_true = results["us_true"]
    scenario_branches = results["scenario_branches"]
    plot_dim = config.plotting_dimension
    #mu_history = results["mu_history"]
    #trace_history = results["trace_history"]

    fig, (ax_xy, ax_u, ax_x) = plt.subplots(1, 3, figsize=(24, 5))

    #Fig 1 - Plot trajectory, scenario branches and safe set

    if  plot_dim != 1 and plot_dim != 2:
        raise ValueError("Unsupported plotting dimension. Must be 1 or 2.")
    
    if plot_dim == 1:
    #Plot safe set
        if config.safety_bounds is not None:
            ax_xy.axhline(y=config.safety_bounds[0], color="tab:red", linewidth=2, linestyle="--")
            ax_xy.axhline(y=config.safety_bounds[1], color="tab:red", linewidth=2, linestyle="--")

        # #Plot scenario branches
        # for branch_idx, branch in enumerate(scenario_branches):
        #     num_scenarios_plotted = min(config.num_scenarios_to_plot, branch.shape[0])
        #     for scenario_idx in range(num_scenarios_plotted):
        #         label = "scenario branches" if (branch_idx == 0 and scenario_idx == 0) else None
        #         ax_xy.plot(
        #             branch[scenario_idx, :, 0],
        #             "--", linewidth=1.0, alpha=0.35, color="tab:orange", label=label
        #         )
        
        #plot trajectory
        t_x = np.arange(len(xs_true))
        ax_xy.plot(t_x, xs_true[:, 0], color="tab:blue", linewidth=3.0, label="true state")
        ax_xy.scatter(0, task.x_init[0], marker="s", s=80, label="start")

        #plot reference trajecrory (or zero trajectory if no reference provided)
        ref_plot = np.array([task.reference_state(k) for k in range(config.num_applied_steps + 1)])
        ax_xy.plot(t_x, ref_plot[:, 0],
            color="tab:green", linewidth=2.0, alpha=0.9, label="reference trajectory"
            )
        
        (x_min, x_max) = config.plot_bounds
        ax_xy.set_ylim(x_min, x_max)
        #ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0.0)

    

    if plot_dim == 2:
    #Plot safe set
        if config.safety_vertices is not None:
            poly = Polygon(config.safety_vertices, closed=True, fill=False, linewidth=2)
            ax_xy.add_patch(poly)
        else:
            #If no vertices but halfspace representation is provided, compute polygon for plotting
            if config.H_p is not None and config.h_p is not None:
                safe_vertices = halfspace_polygon(config.H_p, config.h_p, bbox = config.plot_bounds)
                if safe_vertices.size > 0:
                    poly = Polygon(safe_vertices, closed=True, fill=False, linewidth=2)
                    ax_xy.add_patch(poly)

        #Plot scenario branches
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

         #plot reference trajectory
        ref_plot = np.array([task.reference_state(k) for k in range(config.num_applied_steps + 1)])
        ax_xy.plot(
            ref_plot[:, 0], ref_plot[:,1],
            color="tab:green", linewidth=2.0, alpha=0.9, label="reference trajectory"
        )

        ax_xy.plot(xs_true[:, 0], xs_true[:, 1], color="tab:blue", linewidth=3.0, label="true state")
        ax_xy.scatter(task.x_init[0], task.x_init[1], marker="s", s=80, label="start")
        ax_xy.set_xlabel("x")
        ax_xy.set_ylabel("y")
        (x_min, x_max, y_min, y_max) = config.plot_bounds
        ax_xy.set_aspect("equal", adjustable="box")

        ax_xy.set_xlim(x_min, x_max)
        ax_xy.set_ylim(y_min, y_max)
        ax_xy.grid(True)
        ax_xy.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, borderaxespad=0.0)

    

    if config.use_info_gain:
        ax_xy.set_title("Posterior-aware scenario MPC with Information Gain (modular)")
    else:
        ax_xy.set_title("Posterior-aware scenario MPC without Information Gain (modular)")


    #Fig 2 - Plot control inputs
    color_list = ["tab:blue", "tab:orange", "tab:cyan", "tab:pink", "tab:purple"]
    if len(us_true) > 0:
        t_u = np.arange(len(us_true))
        for i in range(us_true.shape[1]):
            ax_u.step(t_u, us_true[:, i], where="post", label=f"u_{i}", color=color_list[i % len(color_list)])
            if config.u_max is not None:
                ax_u.axhline(config.u_max[i], linestyle="--", linewidth=1.0, color=color_list[i % len(color_list)], alpha=0.7)
                ax_u.axhline(-config.u_max[i], linestyle="--", linewidth=1.0, color=color_list[i % len(color_list)], alpha=0.7)
        ax_u.set_title("Applied control inputs")
        ax_u.set_xlabel("time step")
        ax_u.set_ylabel("u")
        ax_u.grid(True)
        ax_u.legend()

    #Fig 3 - Tracking error vs time
    if len(xs_true) > 0 and len(x_refs) > 0:
        t_x = np.arange(len(xs_true))
        state_labels = [f"x_{d}" for d in range(config.plotting_dimension)]
        for i, label in enumerate(state_labels):
            ax_x.plot(t_x, abs(x_refs[:, i] - xs_true[:, i]), linewidth=1.6, label=f"{label}")
        ax_x.set_title("Tracking Error")
        ax_x.set_xlabel("time step")
        ax_x.set_ylabel("position")
        ax_x.grid(True)
        ax_x.legend(ncol=2)



#Old plotting code

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

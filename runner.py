import numpy as np


def run_simulation(task, cfg, controller, bayes_model, rng):
    x_true = task.x_init.copy()
    xs_true = [x_true.copy()]
    x_refs = [task.reference_state(0).copy()]
    us_true = []
    scenario_branches = []
    mu_history = [bayes_model.mu.copy()]
    trace_history = [float(np.trace(bayes_model.P))]

    x_info_refs = [np.tile(x_true.reshape(task.n_x, 1), (1, cfg.N + 1)) for _ in range(cfg.S)]
    u_info_ref = np.zeros((task.n_u, cfg.N))

    applied_steps = 0
    while applied_steps < cfg.num_applied_steps:
        W_info = bayes_model.posterior_weight_matrix()

        scenario_models = []
        for s in range(cfg.S):
            A_tilde_s, B_s = bayes_model.sample_dynamics(rng)
            scenario_models.append((A_tilde_s, B_s))
            x_info_refs[s] = np.tile(x_true.reshape(task.n_x, 1), (1, cfg.N + 1))

        ref_horizon = np.column_stack(
            [task.reference_state(applied_steps + k) for k in range(cfg.N + 1)]
        )
        u_plan, x_pred = controller.solve(
            x0=x_true,
            ref_horizon=ref_horizon,
            scenario_models=scenario_models,
            W_info=W_info,
            x_info_refs=x_info_refs,
            u_info_ref=u_info_ref,
        )

        for s in range(cfg.S):
            x_info_refs[s] = x_pred[s]
        u_info_ref = u_plan

        branch = np.stack([x_pred[s].T for s in range(cfg.S)], axis=0)
        scenario_branches.append(branch)

        remaining_steps = cfg.num_applied_steps - applied_steps
        controls_to_apply = min(cfg.actions_per_mpc_solve, cfg.N, remaining_steps)
        for k in range(controls_to_apply):
            u = u_plan[:, k].reshape(task.n_u)
            us_true.append(u.copy())

            x_prev = x_true.copy()
            x_true = task.true_step(x_true, u, rng)
            xs_true.append(x_true.copy())
            x_refs.append(task.reference_state(applied_steps + 1).copy())
            bayes_model.posterior_update(x_prev, u, x_true)
            mu_history.append(bayes_model.mu.copy())
            trace_history.append(float(np.trace(bayes_model.P)))

            applied_steps += 1

    return {
        "xs_true": np.asarray(xs_true),
        "x_refs": np.asarray(x_refs),
        "us_true": np.asarray(us_true),
        "scenario_branches": scenario_branches,
        "mu_history": np.asarray(mu_history),
        "trace_history": np.asarray(trace_history),
        "mu_theta_final": bayes_model.mu.copy(),
        "P_theta_final": bayes_model.P.copy(),
    }

import cvxpy as cp
import numpy as np


class ScenarioMPCController:
    def __init__(self, task, cfg):
        self.task = task
        self.cfg = cfg
        self.n_x = task.n_x
        self.n_u = task.n_u
        self.p = self.n_x + self.n_u
        self.p_decision = self.n_x + self.n_u

        self._build_problem()

    def _build_problem(self):
        N = self.cfg.N
        S = self.cfg.S
        m_poly = self.task.H_x_full.shape[0]

        self.U = cp.Variable((self.n_u, N))
        self.X = [cp.Variable((self.n_x, N + 1)) for _ in range(S)]
        self.slack = [cp.Variable((m_poly, N + 1), nonneg=True) for _ in range(S)]
        self.vel_slack = [cp.Variable((2, N + 1), nonneg=True) for _ in range(S)]

        self.x0_param = cp.Parameter(self.n_x)
        self.ref_state_param = cp.Parameter((self.n_x, N + 1))
        self.A_params = [cp.Parameter((self.n_x, self.n_x)) for _ in range(S)]
        self.B_params = [cp.Parameter((self.n_x, self.n_u)) for _ in range(S)]
        self.info_grad_params = [cp.Parameter((self.p_decision, N)) for _ in range(S)]

        constraints = []
        tracking_cost = 0.0
        info_lin_reward = 0.0

        for k in range(N):
            constraints += [
                self.U[:, k] <= self.cfg.u_max,
                self.U[:, k] >= -self.cfg.u_max,
            ]
            tracking_cost += cp.quad_form(self.U[:, k], self.cfg.R)

        for s in range(S):
            constraints += [self.X[s][:, 0] == self.x0_param]
            for k in range(N):
                constraints += [
                    self.X[s][:, k + 1]
                    == self.A_params[s] @ self.X[s][:, k] + self.B_params[s] @ self.U[:, k]
                ]
                constraints += [
                    self.X[s][2:, k] <= self.cfg.v_max + self.vel_slack[s][:, k],
                    self.X[s][2:, k] >= -self.cfg.v_max - self.vel_slack[s][:, k],
                ]
                constraints += [
                    self.task.H_x_full @ self.X[s][:, k] <= self.task.h_p + self.slack[s][:, k]
                ]

                tracking_cost += (1.0 / S) * cp.quad_form(
                    self.X[s][:, k] - self.ref_state_param[:, k], self.cfg.Q
                )
                tracking_cost += (self.cfg.slack_weight / S) * cp.sum(self.slack[s][:, k])
                tracking_cost += (self.cfg.velocity_slack_weight / S) * cp.sum(self.vel_slack[s][:, k])

                info_lin_reward += (1.0 / S) * (
                    cp.sum(cp.multiply(self.info_grad_params[s][:self.n_x, k], self.X[s][:, k]))
                    + cp.sum(cp.multiply(self.info_grad_params[s][self.n_x:, k], self.U[:, k]))
                )

            constraints += [
                self.X[s][2:, N] <= self.cfg.v_max + self.vel_slack[s][:, N],
                self.X[s][2:, N] >= -self.cfg.v_max - self.vel_slack[s][:, N],
                self.task.H_x_full @ self.X[s][:, N] <= self.task.h_p + self.slack[s][:, N],
            ]
            tracking_cost += (1.0 / S) * cp.quad_form(
                self.X[s][:, N] - self.ref_state_param[:, N], self.cfg.Qf
            )
            tracking_cost += (self.cfg.slack_weight / S) * cp.sum(self.slack[s][:, N])
            tracking_cost += (self.cfg.velocity_slack_weight / S) * cp.sum(self.vel_slack[s][:, N])

        total_objective = (
            self.cfg.cost_weight * tracking_cost
            - self.cfg.info_gain_weight * info_lin_reward
        )
        self.problem = cp.Problem(cp.Minimize(total_objective), constraints)

    def solve(
        self,
        x0: np.ndarray,
        ref_horizon: np.ndarray,
        scenario_models,
        W_info: np.ndarray,
        x_info_refs,
        u_info_ref: np.ndarray,
    ):
        self.x0_param.value = x0
        self.ref_state_param.value = ref_horizon

        for s in range(self.cfg.S):
            A_s, B_s = scenario_models[s]
            self.A_params[s].value = A_s
            self.B_params[s].value = B_s

            info_grad = np.zeros((self.p_decision, self.cfg.N))
            for k in range(self.cfg.N):
                psi_ref = np.hstack([x_info_refs[s][:, k], u_info_ref[:, k]])
                grad_full = 2.0 * (W_info @ psi_ref)
                info_grad[:, k] = grad_full
            self.info_grad_params[s].value = info_grad

        self.problem.solve(solver=getattr(cp, self.cfg.solver), **self.cfg.solver_opts)
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC solve failed: {self.problem.status}")

        x_pred = [np.asarray(self.X[s].value) for s in range(self.cfg.S)]
        u_plan = np.asarray(self.U.value)
        return u_plan, x_pred

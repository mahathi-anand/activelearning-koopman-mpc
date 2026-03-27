import numpy as np

from bayes_linear_model import BayesianLinearRegressionDynamics
from config import MPCConfig
from mpc_controller import ScenarioMPCController
from plotting import plot_run
from runner import run_simulation
from task import Figure8GravityTask


def main():
    task = Figure8GravityTask()
    mpc_cfg = MPCConfig()

    bayes_model = BayesianLinearRegressionDynamics(task=task)
    controller = ScenarioMPCController(task=task, cfg=mpc_cfg)

    P_theta_init = bayes_model.P.copy()
    rng = np.random.default_rng(4)
    results = run_simulation(task, mpc_cfg, controller, bayes_model, rng)

    print("Initial trace(P_theta):", np.trace(P_theta_init))
    print("Final trace(P_theta):  ", np.trace(results["P_theta_final"]))
    A_hat_final, _ = bayes_model.unpack_theta(results["mu_theta_final"])
    print("Nominal A error (Fro):", np.linalg.norm(A_hat_final - task.A_nom))


    plot_run(task, mpc_cfg, results)


if __name__ == "__main__":
    main()

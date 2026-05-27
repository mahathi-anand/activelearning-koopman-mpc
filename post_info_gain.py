import numpy as np

from bayes_linear_model import BayesianLinearRegressionDynamics
from config import KoopmanVanDerPolMPCConfig
from mpc_controller import ScenarioMPCController
from plotting import plot_run
from runner import run_simulation
from task import KoopmanVanDerPolTask


def main():
    task = KoopmanVanDerPolTask()
    mpc_cfg = KoopmanVanDerPolMPCConfig()

    bayes_model = BayesianLinearRegressionDynamics(task=task)
    controller = ScenarioMPCController(task=task, cfg=mpc_cfg)

    P_theta_init = bayes_model.P.copy()
    rng = np.random.default_rng(4)
    results = run_simulation(task, mpc_cfg, controller, bayes_model, rng)

    print("Initial trace(P_theta):", np.trace(P_theta_init))
    print("Final trace(P_theta):  ", np.trace(results["P_theta_final"]))
    A_hat_final, _ = bayes_model.unpack_theta(results["mu_theta_final"])
    print("A error vs eval (Fro):", np.linalg.norm(A_hat_final - task.A_eval))

    plot_run(task, mpc_cfg, results)


if __name__ == "__main__":
    main()

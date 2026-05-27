from imports import *
from system import LinearSystem, VanDerPolSystem
from koopman_operator import KoopmanOperator
from config import SimpleMPCConfig, KoopmanVanDerPolConfig
from mpc_controller import ScenarioMPC
from task import ReferenceTask
from bayes_linear_model import BayesianLRDynamics
from runner import run_simulation
from plotting import plot_traj



if __name__ == "__main__":
    original_system = VanDerPolSystem(mu=0.2, dt=0.01)  #Original System Dynamics TODO: Add argument parser to specify system parameters and other configurations
    print(f"System: {original_system.system_name}, State Dimension: {original_system.n_x}, Control Dimension: {original_system.n_u}")

    #Compute Koopman Operator from Data - Offline Step
    koopman_operator = KoopmanOperator(original_system)
    data = pd.read_csv('data/' + original_system.system_name + '_damp_' + str(original_system.mu) + '.csv')
    X = data[['x1', 'x2']].values.T
    U = data[['u']].values.T
    X_plus = data[['x1_plus', 'x2_plus']].values.T
    print(f"Data Loaded: {X.shape[1]} samples... Computing Koopman Operator...")
    [A, B] = koopman_operator.compute_operator(X, U, X_plus)

    print(f"Koopman Operator Computed: A shape={A.shape}, B shape={B.shape}")
    #Koopman System
    koopman_system = LinearSystem(A, B, name="VanDerPol")

    #Simulate both systems to compare trajectories
    num_steps = 2000
    rows = 2
    columns = 2
    fig, axes = plt.subplots(rows, columns, figsize=(10, 10))
    fig.suptitle('Comparison of Nonlinear and Koopman System Trajectories - ' + original_system.system_name + ' Damping Factor =' + str(original_system.mu), fontsize=16)

    for u in range(rows * columns):
        ax = plt.subplot(2,2, u+1)
        x0 = np.random.uniform(-5, 5, size=(2,))
        #Simulate original system
        u_seq = np.random.uniform(-2, 2, size=(1, num_steps))
        traj_vanderpol = original_system.simulate(x0, u_seq.T)

        #Simulate Koopman system
        z0, _ = original_system.observables(x0)
        traj_koopman = koopman_system.simulate(z0, u_seq.T)
        traj_koopman_ext = traj_koopman[:, :2] #Extracting the original state dimensions from the Koopman trajectory

        #Plotting trajectories  
        ax.plot(traj_vanderpol[:, 0], traj_vanderpol[:, 1], label='Van der Pol', color='blue')
        ax.plot(traj_koopman[:, 0], traj_koopman[:, 1], label='Koopman', color='blue', linestyle='dashed')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()

    plt.tight_layout()
    plt.show()

    #Scenario MPC - Online Step with Iterative Computation of Koopman Operator
    
    #configure the task
    x_phys_init = np.random.uniform(-3, 3, size=(2,))  # physical initial state
    x_init = original_system.observables(x_phys_init)[0]
    task = ReferenceTask(system = koopman_system, prior_mean = 0.0, x_init = x_init, nonlinear_system = original_system)
    mpc_config = KoopmanVanDerPolConfig(system = koopman_system, use_info_gain = True, num_applied_steps = 500)
    bayes_model = BayesianLRDynamics(system=koopman_system, task=task)
    controller = ScenarioMPC(task = task, config = mpc_config)

    #Initial logging
    P_theta_init = bayes_model.P.copy()
    rng = np.random.default_rng(4)
    results =  run_simulation(task, mpc_config, controller, bayes_model, rng)

    print("Initial trace(P_theta):", np.trace(P_theta_init))
    print("Final trace(P_theta):  ", np.trace(results["P_theta_final"]))
    A_hat_final, _ = bayes_model.unpack_theta(results["mu_theta_final"])

    plot_traj(task, mpc_config, results)






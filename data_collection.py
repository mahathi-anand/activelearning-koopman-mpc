import numpy as np
import pandas as pd
import os
from system import VanDerPolSystem


def collect_data(system, N_data, state_range, control_range):
    x = state_range[0] + (state_range[1] - state_range[0]) * np.random.rand(2, N_data)
    u = control_range[0] + (control_range[1] - control_range[0]) * np.random.rand(1, N_data)
    x_plus = np.zeros_like(x)
    for i in range(N_data):
        x_plus[:, i] = system.dynamics(x[:, i], u[:, i])
    return x, u, x_plus


if __name__ == "__main__":
    system = VanDerPolSystem(mu=0, dt=0.01)
    N_data = 50000
    state_range = [-5,5]
    control_range = [-2, 2]
    x, u, x_plus = collect_data(system, N_data, state_range, control_range)

    #Save data to csv
    df = pd.DataFrame({
        'x1': x[0, :],
        'x2': x[1, :],
        'u': u[0, :],
        'x1_plus': x_plus[0, :],
        'x2_plus': x_plus[1, :]
    })
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/' + system.system_name + '.csv', index=False)









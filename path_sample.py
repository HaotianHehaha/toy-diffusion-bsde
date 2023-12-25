import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(dimention, n_steps, dt):
    """
    Simulate Brownian motion path.

    Parameters:

    - dimension: sampling path number and dimention
    - n_steps: Number of steps in the simulation.
    - dt: Time step size.

    Returns:
    - path: Numpy array representing the increments of Brownian motion path.
    """
    path_num = dimention[0]
    variable_num = dimention[1]
    increments = np.zeros((path_num,n_steps,variable_num))
    for i in range(path_num):
        mean = np.zeros((variable_num,))
        cov = dt*np.eye(variable_num)
        increments[i] = np.random.multivariate_normal(mean, cov, n_steps)

    return increments

def plot_brownian_motion(path, dt):
    """
    Plot the Brownian motion path.

    Parameters:
    - path: Numpy array representing the Brownian motion path.
    - dt: Time step size.
    """
    t = np.arange(0, len(path)) * dt
    plt.plot(t, path)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Brownian Motion Path')
    plt.show()

# Set the number of steps and time step size
# n_steps = 1000
# dt = 0.001

# Generate Brownian motion path
#path = brownian_motion(n_steps, dt)


import numpy as np

from utils.target_function import surface_function
from utils.particle_swarm import ParticleSwarm
from utils.matplotlib_animator import ScatterPlotAnimator

# setting the global seed
np.random.seed(42)

# vectorizing target function (so it could work with numpy arrays)
target_function = np.vectorize(surface_function)

# defining simulation hyperarameters
n_particles = 36
n_iterations = 100

c1, c2 = 2, 2  # cognitive and social component coefficiants
w_min, w_max = 0.4, 0.9  # bounds for the inertia coefficiant
v_max = 6  # max possible velocity (to cap speeding particles)

particle_seed_x_bounds = (-5, 5)  # area bounds for seeding the particles on the x axis
particle_seed_y_bounds = (-5, 5)  # area bounds for seeding the particles on the y axis

random_initialisation = False  # whether to plant particles in a random or grid fashion
position_cap = True     # if True, will not allow particles to venture outside planting x, y bounds
                        # if False, some particles may go into -+Inf and crash

velocity_cap = True  # if True, will not allow particles velocity to exceed +- v_max
                     # if False, some particles may wander very long due to large inertia

# defining visualisation hyperparameters
title = f"w = [{w_min}, {w_max}], c1 = {c1}, c2 = {c2}"

plot_x_bounds = (-5, 5)  # x axis bounds for the plot
plot_y_bounds = (-5, 5)  # y axis bounds for the plot

frame_interval = 48  # time interval between frames (smaller = faster animation)


# making the simulation object that will generate particle locations for each iteration
swarm_simulation = ParticleSwarm(
    target_function=target_function,
    n_particles=n_particles,
    c1=c1,
    c2=c2,
    w_min=w_min,
    w_max=w_max,
    v_max=v_max,
    n_iterations=n_iterations,
    x_bounds=particle_seed_x_bounds,
    y_bounds=particle_seed_y_bounds,
    random_initialisation=random_initialisation,
    position_cap=position_cap,
    velocity_cap=velocity_cap,
)

# making Animator object that will animate the scatter plot
animation = ScatterPlotAnimator(
    title=title,
    target_function=target_function,
    x_bounds=plot_x_bounds,
    y_bounds=plot_y_bounds,
    simulation_object=swarm_simulation,
    frame_n=n_iterations,
    frame_interval=frame_interval,
)

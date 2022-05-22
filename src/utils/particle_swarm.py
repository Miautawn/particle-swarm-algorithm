from math import sqrt, floor, ceil
import numpy as np


class ParticleSwarm:
    """
    Simulates the particle swarm optimisation.
    """

    def __init__(
        self,
        target_function,
        n_particles,
        c1,
        c2,
        w_min,
        w_max,
        v_max,
        n_iterations,
        x_bounds,
        y_bounds,
        random_initialisation=True,
        position_cap=True,
        velocity_cap=True,
    ):

        self.target_function = target_function

        self.n_particles = n_particles
        self.n_iterations = n_iterations

        self.c1, self.c2 = c1, c2              # cognitive and social component coefficiants
        self.w_min, self.w_max = w_min, w_max  # bounds for inertia weight
        self.w = w_max                         # starting intertia weight
        self.v_max = v_max                     # max velocity
        self.x_bounds, self.y_bounds = (
            x_bounds,
            y_bounds,
        )                                      # lower and upper bounds for the x and y axies

        self.particles_x = np.empty(
            [n_particles, 2]
        )                                      # vector of particle locations [n_particles, 2]
        self.particles_v = np.empty(
            [n_particles, 2]
        )                                      # vector of particle velocities [n_particles, 2]
        self.particles_pb = np.empty(
            [n_particles, 2]
        )                                      # vector of coordinates for personal bests [n_particles, 2]
        self.particles_gb = np.empty([2])      # vector of coordinates for global best [2]

        self.particles_x_target = np.empty(
            [n_particles]
        )                                      # vector of values for current positions [n_particles,]
        self.particles_pb_target = np.empty(
            [n_particles]
        )                                      # vector of values for personal bests [n_particles,]
        self.particles_gb_target = None        # a single value for global best value

        self.position_cap = position_cap       # if True, will not allow particles to venture outside planting x, y bounds
        self.velocity_cap = velocity_cap       # if True, will not allow particles velocity to exceed -+ v_max (on any axis)

        self.initialize_particles(x_bounds, y_bounds, random_initialisation)

    def initialize_particles(self, x_bounds, y_bounds, random_initialisation):
        """
        initiate the particles:
        1.) Planst the particles either randomly or in the uniform grid
        2.) Randomly sets their velocity vectors
        3.) Randomly sets their personal best points
        4.) Randomly sets the global best point
        """
        # setting the initial particle locations
        if random_initialisation:
            self.particles_x[:, 0] = np.random.uniform(*x_bounds, size=self.n_particles)
            self.particles_x[:, 1] = np.random.uniform(*y_bounds, size=self.n_particles)
        else:

            # initially plant all the points in the middle
            self.particles_x[:] = np.array([[sum(x_bounds) / 2, sum(y_bounds) / 2]])

            # will plant this^2 ammount of particles in a uniform grid
            max_full_grid_dim = floor(sqrt(self.n_particles))

            self.particles_x[: max_full_grid_dim**2] = np.array(
                [
                    [i, j]
                    for i in np.linspace(*x_bounds, max_full_grid_dim)
                    for j in np.linspace(*y_bounds, max_full_grid_dim)
                ]
            )

        # setting the initial velocity vectors
        self.particle_v = np.random.rand(self.n_particles, 2)

        # setting the initial personal best points & values
        self.particles_pb = np.random.rand(self.n_particles, 2) * self.v_max
        self.particles_pb_target[:] = np.Inf

        # setting the initial global best point
        self.particles_gb = np.zeros(2)
        self.particles_gb_target = np.Inf

    def sample_iteration(self):
        """
        Return the current position of the particles
        """

        return self.particles_x

    def simulate(self):
        """
        Runs the simulation for self.n_iterations,
        Works as a python generator and yields particle positions each iteration
        """
        for iteration_n in range(1, self.n_iterations + 1):

            # updating target values for each particle
            self.particle_x_target = self.target_function(
                self.particles_x[:, 0], self.particles_x[:, 1]
            )

            # updating personal bests
            personal_best_mask = self.particle_x_target < self.particles_pb_target
            self.particles_pb_target[personal_best_mask] = self.particle_x_target[
                personal_best_mask
            ]
            self.particles_pb[personal_best_mask] = self.particles_x[personal_best_mask]

            # updating global best
            iteration_lowest_value_index = np.argmin(self.particles_pb_target)
            if (
                self.particles_pb_target[iteration_lowest_value_index]
                < self.particles_gb_target
            ):
                self.particles_gb_target = self.particles_pb_target[
                    iteration_lowest_value_index
                ]
                self.particles_gb = self.particles_pb[iteration_lowest_value_index]

            # updating the inertia weight linearly
            self.w = self.w_max - iteration_n * (
                (self.w_max - self.w_min) / self.n_iterations
            )

            # updating particle velocity vectors
            self.particles_v = (
                self.w * self.particles_v                 # inertia
                + self.c1
                * np.random.rand()
                * (self.particles_pb - self.particles_x)  # cognitive component
                + self.c2
                * np.random.rand()
                * (self.particles_gb - self.particles_x)  # social component
            )

            # bouding velocity to v_max and -v_max
            if self.velocity_cap:
                overboard_velocity_masks = (
                    self.particles_v > self.v_max,
                    self.particles_v < -self.v_max,
                )

                self.particles_v[overboard_velocity_masks[0]] = (
                    np.random.rand(np.sum(overboard_velocity_masks[0])) * self.v_max
                )
                self.particles_v[overboard_velocity_masks[1]] = (
                    np.random.rand(np.sum(overboard_velocity_masks[1])) * -self.v_max
                )

            # updating particle position vectors
            self.particles_x = self.particles_x + self.particles_v

            # bouding the points to the search space
            if self.position_cap:
                self.particles_x[:, 0][
                    self.particles_x[:, 0] > self.x_bounds[1]
                ] = self.x_bounds[1]
                self.particles_x[:, 0][
                    self.particles_x[:, 0] < self.x_bounds[0]
                ] = self.x_bounds[0]

                self.particles_x[:, 1][
                    self.particles_x[:, 1] > self.y_bounds[1]
                ] = self.y_bounds[1]
                self.particles_x[:, 1][
                    self.particles_x[:, 1] < self.y_bounds[0]
                ] = self.y_bounds[0]

            yield self.particles_x

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation as plt_animation


class ScatterPlotAnimator:
    def __init__(
        self,
        target_function,
        x_bounds,
        y_bounds,
        simulation_object,
        frame_n,
        title="",
        frame_interval=50,
    ):

        self.target_function = target_function  # vectorized target function

        self.frame_n = frame_n  # how many frames to render
        self.frame_interval = frame_interval  # time interval between frames

        self.simulation_object = simulation_object  # object that generates point data
        self.title = title  # Title string for the plot

        self.x_bounds = x_bounds  # x bounds for the plot
        self.y_bounds = y_bounds  # y bounds for the plot
        self.figure, self.axis = plt.subplots(figsize=(10, 10))

        # adding initial points on the plot
        points = self.simulation_object.sample_iteration()
        self.scatter = self.axis.scatter(points[:, 0], points[:, 1], s=8, c="black")

        self.animation = plt_animation.FuncAnimation(
            self.figure,
            self.update,
            interval=self.frame_interval,
            frames=self.simulation_object.simulate(),
            save_count=self.frame_n,
            init_func=self.setup_plot,
            blit=True,
        )

    def setup_plot(self):
        """
        Setups the initial visual plot
        """
        # set the contour plot
        x = np.linspace(*self.x_bounds, 100)
        y = np.linspace(*self.y_bounds, 100)

        xx, yy = np.meshgrid(x, y)
        zz = self.target_function(xx, yy)

        self.axis.contour(xx, yy, zz, levels=15)

        # name the axis
        self.axis.set_xlabel("x")
        self.axis.set_ylabel("y")

        # add title
        self.axis.set_title(self.title)

        return (self.scatter,)

    def update(self, points):
        """
        Updates the plot by re-drawing points
        """
        self.scatter.set_offsets(points)

        return (self.scatter,)

    def render(self):
        """
        Renders the animation to be displayed in GUI
        """
        return plt.show()

import matplotlib.pyplot as plt

from beavr.teleop.configs.constants import robots

from .plotter import Plotter


class PlotHand3D(Plotter):
    def __init__(self):
        # Initializing the figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Loading Joint information
        self.joint_information = robots.OCULUS_JOINTS
        self.view_limits = robots.OCULUS_VIEW_LIMITS

        # Setting the visualizer limits
        self._set_limits()

    def _plot_line(self, x1, x2, y1, y2, z1, z2):
        self.ax.plot([x1, x2], [y1, y2], [z1, z2])

    def _set_limits(self):
        self.ax.set_xlim(self.view_limits["x_limits"])
        self.ax.set_ylim(self.view_limits["y_limits"])
        self.ax.set_zlim3d(self.view_limits["z_limits"][0], self.view_limits["z_limits"][1])

    def _draw_hand(self, x, y, z):
        self.plot3D = self.ax.scatter3D(x, y, z)

        # Drawing connections fromn the wrist - 0
        for idx in self.joint_information["metacarpals"]:
            self._plot_line(x[0], x[idx], y[0], y[idx], z[0], z[idx])

        # Drawing knuckle to knuckle connections and knuckle to finger connections
        for key in ["knuckles", "thumb", "index", "middle", "ring", "pinky"]:
            for idx in range(len(self.joint_information[key]) - 1):
                self._plot_line(
                    x[self.joint_information[key][idx]],
                    x[self.joint_information[key][idx + 1]],
                    y[self.joint_information[key][idx]],
                    y[self.joint_information[key][idx + 1]],
                    z[self.joint_information[key][idx]],
                    z[self.joint_information[key][idx + 1]],
                )

    def draw(self, x, y, z):
        # Setting plotting limits
        self._set_limits()

        # Plotting the hand bones
        self._draw_hand(x, y, z)
        plt.draw()

        # Resetting and Pausing the 3D plot
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        plt.cla()

        # Removing the drawing
        self.plot3D.remove()


class PlotHandDirection(Plotter):
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        self._set_limits()

    def _set_limits(self):
        self.ax.set_xlim([-0.3, 0.3])
        self.ax.set_ylim([-0.3, 0.3])
        self.ax.set_zlim3d(-0.3, 0.3)

    def draw(self, x, y, z):
        self._set_limits()
        self.plot3D = self.ax.scatter(x, y, z)

        # Draw the axes
        self.ax.plot([x[0], x[1]], [y[0], y[1]], [z[0], z[1]], color="blue", label="hand_cross")
        self.ax.plot([x[0], x[2]], [y[0], y[2]], [z[0], z[2]], color="green", label="hand_normal")
        self.ax.plot(
            [x[0], x[3]],
            [y[0], y[3]],
            [z[0], z[3]],
            color="red",
            label="hand_direction",
        )

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        plt.draw()

        # Resetting and Pausing the 3D plot
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        plt.cla()

        # Removing the drawing
        self.plot3D.remove()

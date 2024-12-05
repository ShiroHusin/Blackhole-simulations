import numpy as np
import matplotlib.pyplot as plt


class Accretion_disc:
    def __init__(self, Rs, num_particles=1000, disc_thickness=0.1, scale=2.5):
        self.R_photon = 1.5 * Rs
        self.Rs = Rs
        self.num_particles = num_particles
        self.disc_thickness = disc_thickness
        self.scale = scale

    def generate_coordinates(self):

        r = np.random.exponential(scale=self.scale, size=self.num_particles) + self.R_photon
        theta = np.random.normal(loc=np.pi / 2, scale=self.disc_thickness, size=self.num_particles)
        phi = np.random.uniform(0, 2 * np.pi, size=self.num_particles)

        return r, theta, phi

    def boyer_lindquist_to_cartesian(self, r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def plot_disc(self):
        r, theta, phi = self.generate_coordinates()
        x, y, z = self.boyer_lindquist_to_cartesian(r, theta, phi)

        fig = plt.figure(figsize=(15, 6))

        # X-Y plane (top view)
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(x, y, z, s=1, c='orange', alpha=0.6)
        ax1.set_title('3D View')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # X-Y plane (top view)
        ax2 = fig.add_subplot(132)
        ax2.scatter(x, y, s=1, c='blue', alpha=0.6)
        ax2.set_title('Top View (X-Y)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.axis('equal')

        # X-Z plane (side view)
        ax3 = fig.add_subplot(133)
        ax3.scatter(x, z, s=1, c='red', alpha=0.6)
        ax3.set_title('Side View (X-Z)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.axis('equal')

        plt.tight_layout()
        plt.show()

Rs = 2.0  # Schwarzschild radius
accretion_disc = Accretion_disc(Rs, num_particles=10000, disc_thickness=0.05)
accretion_disc.plot_disc()

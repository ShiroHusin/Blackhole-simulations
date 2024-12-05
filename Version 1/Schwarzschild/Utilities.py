import numpy as np
from scipy.interpolate import interp1d

def spherical_to_cartesian(theta, phi, r = 1):
    """Convert spherical coordinates to cartesian."""
    x = r*np.sin(theta) * np.cos(phi)
    y = r*np.sin(theta) * np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z

def cartesian_to_spherical(x, y, z, camera_position):
    """Convert 3D cartesian points into a spherical coordinate system"""
    r = np.sqrt((x - camera_position[0])**2 + (y - camera_position[1])**2 + (z - camera_position[2])**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y - camera_position[1], x - camera_position[0])
    return r, theta, phi

def basic_deflection_angle(Rs, b):
    """Calculate the basic deflection angle due to gravity.
        This is the Phi angle the azimuthal angle.
        phi = -4 * Gm/c^2 b so -2 * Rs/b
    """
    return -2 * Rs / b

def deviated_angle(Phi, D, R):
    """Calculate the deviated angle with additional correction."""

    return Phi + np.arcsin(D / R * np.sin(Phi))

def rotation_matrix(beta):
    """
    This is involved in rotating the camera to view the black hole and the matrix involved in it.
    This matrix rotates all the points along Rx. Since the angles are measured from the top, We need
    to logic some parts of beta.
    """

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(beta), -np.sin(beta)],
                   [0, np.sin(beta), np.cos(beta)]])

    return Rx

def interpolate(x_pivot, f_pivot, kind='cubic'):
    """Create interpolation data to reduce computation time."""
    interpolation = interp1d(x_pivot, f_pivot,
                             kind=kind, bounds_error=False)
    return interpolation

def adjust_phi(phi, degrees = False):
    """
    Adjust the phi angle.
    If phi is above 90 degrees, subtract by 90 degrees (pi/2 radians).
    If negative, add 90 degrees (pi/2 radians).
    """
    if not degrees:
        phi = np.where(phi > np.pi/2, phi - np.pi/2, phi)
        phi = np.where(phi < 0, phi + np.pi/2, phi)
        return phi.item()
    else:
        phi = np.where(phi > 90, phi - 90, phi)
        phi = np.where(phi < 0, phi + 90, phi)
        return phi.item()

def create_circle_points(center, radius, num_points=8):
    """Generate points on the circumference of a circle in the Y-Z plane."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = []
    for angle in angles:
        y = radius * np.cos(angle)
        z = radius * np.sin(angle)
        points.append([center[0], y, z])
    return np.array(points)

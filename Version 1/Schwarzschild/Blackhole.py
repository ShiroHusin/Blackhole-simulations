from Utilities import *
from tqdm import trange
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp  # Integrate ord diff eqs
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from PIL import Image




class black_hole():
    def __init__(self, schwarzschild_radius, photon_shooter_loc, viewport_loc, image_path, resize_length, fov_side, fov_above):
        ## Physics parts
        self.Rs = schwarzschild_radius
        self.R_isco = 1.5 * self.Rs
        self.photon_loc = photon_shooter_loc
        self.Dist = viewport_loc
        self.limits = self.photon_loc * 2
        self.viewport_loc = viewport_loc

        ## Ray tracing parts
        self.image_path = image_path
        self.resize_length = resize_length
        self.fov_side = fov_side
        self.fov_above = fov_above
        self.image_bounds =  50
        image = self.open_image()
        self.image_width, self.image_height = image.size

    def photon_geodesic_eq(self, phi, u):
        """Represent the differential equation: d²u(ɸ)/dɸ² = 3/2 * Rs * u²(ɸ) - u(ɸ) by splitting it into 2 first order ODEs"""
        v0 = u[1]  # corresponds to u'
        v1 = 3 / 2 * self.Rs * u[0] ** 2 - u[0]  # corresponds to u"
        return v0, v1

    def stop_(self, phi, u):
        """Stop solver if radius < black hole ISCO"""
        with np.errstate(all='ignore'):
            return 1 / u[0] - self.Rs

    stop_.terminal = True

    def solver(self, alpha, dist, num_points=8000):
        """Solve the differential equation for a static black hole using solve_ivp.
            This function works in degrees.
            Due to how the ray tracing is shaping up it can only take positive or negative obtuse angles
        """
        max_angle = 20 * np.pi
        if alpha == 0:
            r = [0]
            phi = [0]
            return r, phi
        elif alpha == np.pi:
            r = [dist]  # Return the original distance of the camera
            phi = [0]  # Return the phi as 180 because there is literally no deviation
            return r, phi
        else:
            if -np.pi < alpha < -np.pi / 2:
                alpha = alpha + np.pi
                y0 = [1 / dist, 1 / (dist * np.tan(alpha))]
                t_eval = np.linspace(0, max_angle, num_points)
                sol = solve_ivp(fun=self.photon_geodesic_eq, t_span=[0, max_angle], y0=y0, method='Radau', t_eval=t_eval,
                                events=self.stop_)
                phi = np.array(sol.t)
                r = np.abs(1 / sol.y[0, :])
                return r, -phi

            elif np.pi / 2 < alpha < np.pi:
                alpha = np.pi - alpha
                y0 = [1 / dist, 1 / (dist * np.tan(alpha))]
                t_eval = np.linspace(0, max_angle, num_points)
                sol = solve_ivp(fun=self.photon_geodesic_eq, t_span=[0, max_angle], y0=y0, method='Radau', t_eval=t_eval,
                                events=self.stop_)
                phi = np.array(sol.t)
                r = np.abs(1 / sol.y[0, :])
                return r, phi

            else:

                y0 = [1 / dist, 1 / (dist * np.tan(alpha))]
                t_eval = np.linspace(0, max_angle, num_points)
                sol = solve_ivp(fun=self.photon_geodesic_eq, t_span=[0, max_angle], y0=y0, method='Radau', t_eval=t_eval,
                                events=self.stop_)
                phi = np.array(sol.t)
                r = np.abs(1 / sol.y[0, :])
                return r, phi

    def check_plot(self, angle_to_consider):
        '''
            Function simply used to check if things make sense so far.
        '''

        r, phi = self.solver(angle_to_consider,self.photon_loc, num_points=8000)
        boole = r < self.limits
        r = r[boole]

        phi = phi[boole]
        x, y, z = spherical_to_cartesian(np.pi / 2, phi, r)
        # Plotting
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 5))
        z = np.zeros_like(z)  # Because of python having 0 being stored into really tiny numbers instead I wrote that
        # Plot in Cartesian coordinates (3D)
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot3D(x, y, z)
        ax1.set_title('Cartesian Coordinates (X, Y, Z)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim(-self.limits / 2, self.limits / 2)
        ax1.set_ylim(-self.limits / 2, self.limits / 2)
        ax1.set_zlim(-self.limits / 2, self.limits / 2)
        ## Add the BH
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_black = self.Rs * np.outer(np.cos(u), np.sin(v))
        y_black = self.Rs * np.outer(np.sin(u), np.sin(v))
        z_black = self.Rs * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x_black, y_black, z_black, color='black', label='blackhole')

        # Plot in polar coordinates
        ax2 = fig.add_subplot(122, projection='polar')
        ax2.plot(phi, r)  # Swapped the order of phi and r
        ax2.set_title('Polar Coordinates (r, phi)')
        ax2.set_theta_zero_location('E')  # Set 0 degrees to the top

        plt.tight_layout()
        plt.show()

    def check_plot_multiples(self, angles_list, return_array=False):
        '''
            Function to check if things make sense so far for multiple angles.
        '''
        all_r = []
        all_phi = []
        all_coords = []

        for angle in angles_list:
            r, phi = self.solver(angle, self.photon_loc, num_points=8000)
            boole = r < self.limits
            r = r[boole]
            phi = phi[boole]
            all_r.append(r)
            all_phi.append(phi)

            if return_array:
                x, y, z = spherical_to_cartesian(np.pi / 2, phi, r)
                z = np.zeros_like(z)
                all_coords.append((x, y, z))

        if return_array:
            return all_coords

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 5))

        # Plot in Cartesian coordinates (3D)
        ax1 = fig.add_subplot(121, projection='3d')
        for r, phi in zip(all_r, all_phi):
            x, y, z = spherical_to_cartesian(np.pi / 2, phi, r)
            z = np.zeros_like(z)  # Because of python having 0 being stored into really tiny numbers
            ax1.plot3D(x, y, z, color='w')
        ax1.set_title('Cartesian Coordinates (X, Y, Z)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim(-self.limits / 2, self.limits / 2)
        ax1.set_ylim(-self.limits / 2, self.limits / 2)
        ax1.set_zlim(-self.limits / 2, self.limits / 2)
        ## Add the BH
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_black = self.Rs * np.outer(np.cos(u), np.sin(v))
        y_black = self.Rs * np.outer(np.sin(u), np.sin(v))
        z_black = self.Rs * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x_black, y_black, z_black, color='black')

        # Plot in polar coordinates
        ax2 = fig.add_subplot(122, projection='polar')
        for r, phi in zip(all_r, all_phi):
            ax2.plot(phi, r, color='w')  # Swapped the order of phi and r
        ax2.set_title('Polar Coordinates (r, phi)')
        ax2.set_theta_zero_location('E')  # Set 0 degrees to the top

        plt.tight_layout()
        plt.show()

    def refine_alpha(self, lower_bound, upper_bound, gradation):
        """Helper function to refine the alpha value within the given bounds."""
        for angle in np.arange(lower_bound, upper_bound, gradation):
            photon_radii = self.solver(angle, self.photon_loc)[0]
            if photon_radii[-1] > self.R_isco:
                return angle
        return upper_bound  # Return the upper bound if no angle found in the range

    def find_minimum_deviation(self, sig_fig=5):
        """ We need this function to determine the values of alpha where a photon
            with backwards ray tracing gets captured by the blackhole. This is an
            iterative process that needs to be done.
        """
        # Initial coarse search
        gradation = 1
        upper_bound = 0

        for angle in np.arange(1, 180, gradation):
            photon_radii = self.solver(angle, self.photon_loc)[0]
            if photon_radii[-1] > self.R_isco:
                upper_bound = angle
                break

        # Refine search with increasing precision
        for _ in range(sig_fig):  # Refine three times for increasing significant figures
            lower_bound = upper_bound - gradation
            upper_bound = self.refine_alpha(lower_bound, upper_bound, gradation / 10)
            gradation /= 10

        alpha_min = upper_bound
        return alpha_min

    def open_image(self):
        im = Image.open(self.image_path)
        im_length, im_width = im.size
        aspect_ratio = im_length / im_width
        resized_width = int(1 / aspect_ratio * self.resize_length)

        final_image = im.resize((self.resize_length, resized_width), Image.LANCZOS)
        return final_image

    def interpolate_limit(self, x_coords, y_coords, z_coords, x_limit):
        # Check if any x_coords are less than or equal to x_limit
        idx_below_limit = np.where(x_coords <= x_limit)[0]

        if len(idx_below_limit) == 0:
            # No points are below the x_limit, the ray does not reach the limit
            return np.nan, np.nan, np.nan

        else:
            points = np.vstack((x_coords, y_coords, z_coords)).T
            values = np.vstack((y_coords, z_coords)).T

            # Perform interpolation
            result = griddata(points[:, 0], values, x_limit, method='linear', fill_value=np.nan)

            if np.isnan(result).any():
                return np.nan, np.nan, np.nan

            return x_limit, result[0], result[1]

    def compute_nearest_pixel(self, row, original_coords):
        '''
            Placeholder function to compute and identify pixels
            that are the nearest to the ray traced pixel.
        '''

        distances = np.sqrt((original_coords[:, 0] - row['X_geo_limit']) ** 2 +
                            (original_coords[:, 1] - row['Y_geo_limit']) ** 2 +
                            (original_coords[:, 2] - row['Z_geo_limit']) ** 2)
        nearest_idx = np.argmin(distances)
        return nearest_idx

    def rotate_and_process(self, distance_from_bh = 8):
        ## Define the camera
        camera_position = [self.photon_loc, 0, 0]

        # Create a meshgrid for Y and Z centered around 0
        self.aspect_ratio = self.image_height / self.image_width
        # Set the bounds for Y and Z
        y_min, y_max = -self.image_bounds * self.Rs, self.image_bounds * self.Rs
        z_min, z_max = -self.image_bounds * self.Rs, self.image_bounds * self.Rs

        y = np.linspace(y_min * self.aspect_ratio, y_max * self.aspect_ratio, self.image_height)
        z = np.linspace(z_min , z_max, self.image_width)
        Y, Z = np.meshgrid(z, y)
        X = np.full_like(Y, -distance_from_bh * self.Rs)

        spherical_coords = np.array([cartesian_to_spherical(X[i, j], Y[i, j], Z[i, j], camera_position) for i in range(len(Y)) for j in range(len(Y[i]))])
        df = pd.DataFrame(spherical_coords, columns=['r', 'theta', 'phi'])


        cartesian_coords = np.array([spherical_to_cartesian(theta, phi, r) for r, theta, phi in
                                     zip(df['r'], df['theta'], df['phi'])])

        # Add Cartesian coordinates to the DataFrame and rotation angles
        df['X_no_bh'] = cartesian_coords[:, 0]
        df['Y_no_bh'] = cartesian_coords[:, 1]
        df['Z_no_bh'] = cartesian_coords[:, 2]

        df['β'] = - np.arctan(df['Z_no_bh']/df['Y_no_bh'])

        X_no_bh = df['X_no_bh'].values
        Y_no_bh = df['Y_no_bh'].values
        Z_no_bh = df['Z_no_bh'].values
        beta = df['β'].values

        rotated_coords = np.zeros((len(df), 3))
        # Apply the rotation matrix to each set of coordinates
        for i in trange(len(df), desc='Rotating coordinates'):
            coords = np.array([X_no_bh[i], Y_no_bh[i], Z_no_bh[i]])
            rotation_matrix_x = rotation_matrix(beta[i])
            rotated_coords[i] = np.dot(rotation_matrix_x, coords)

        ## Create flags on what to compute and what not to compute based on minimum angle after squashing.
        df['X_no_bh_rot'] = rotated_coords[:, 0]
        df['Y_no_bh_rot'] = rotated_coords[:, 1]
        df['Z_no_bh_rot'] = rotated_coords[:, 2]

        ## Compute all the phi angles by rotating ''X_no_bh_rot'', 'Y_no_bh_rot' and 'Z_no_bh_rot' into spherical coordinates
        # and use that to solve for the geodesic equation. But mark which ones to solve by filtering it out.

        spherical_rot_coords = np.array(
            [cartesian_to_spherical(x, y, z, camera_position) for x, y, z in rotated_coords])

        df['r_rot'] = spherical_rot_coords[:, 0]
        df['theta_rot'] = spherical_rot_coords[:, 1]
        df['phi_rot'] = spherical_rot_coords[:, 2]

        # Filter based on phi angles
        phi_threshold = self.find_minimum_deviation()  # Adjust this threshold as needed
        df['valid'] = np.pi - np.abs(df['phi_rot'])   > phi_threshold

        ## Solve the geodesic equation for all the phi and r angles marked as 'True'.
        #  For each of the phi angles, we have to rotate the datapoints back and limit it to
        #  when the X_coordinate reaches a limit. Use linear interpolation to do that.
        #  But don't use the dataframe to store data. Only use the dataframe to store it once these
        #  operations are done

        df['X_geo_limit'] = np.nan
        df['Y_geo_limit'] = np.nan
        df['Z_geo_limit'] = np.nan

        X_limit = -distance_from_bh * self.Rs - camera_position[0]  ## Measure from camera.

        for index, row in tqdm(df[df['valid']].iterrows(), total=df[df['valid']].shape[0], desc= 'Calculating geodesics'):
            r_solver, phi_solver = self.solver(row['phi_rot'], self.photon_loc)
            x_geo, y_geo, z_geo = spherical_to_cartesian(np.pi / 2, phi_solver, r_solver)
            beta_angle = row['β']
            rotation_matrix_inverse = rotation_matrix(-beta_angle)
            x_geo_rotated,y_geo_rotated, z_geo_rotated \
                = np.dot(rotation_matrix_inverse, np.array([x_geo, y_geo, z_geo]))

            x_geo_limit, y_geo_limit, z_geo_limit = \
                self.interpolate_limit(x_geo_rotated,y_geo_rotated, z_geo_rotated, X_limit)

            # Do a check on the y_geo_limit and z_geo_limit, whether both is within the image bounds meshgrid.
            # If any one of them is beyond the image bounds meshgrid return np.nan for all 3 coordinates.


            if ((y_min <= y_geo_limit <= y_max) and (z_min <= z_geo_limit <= z_max)):
                x_geo_limit, y_geo_limit, z_geo_limit = x_geo_limit, y_geo_limit, z_geo_limit
            else:
                x_geo_limit, y_geo_limit, z_geo_limit = np.nan, np.nan, np.nan


            df.at[index, 'X_geo_limit'] = x_geo_limit
            df.at[index, 'Y_geo_limit'] = y_geo_limit
            df.at[index, 'Z_geo_limit'] = z_geo_limit

        df.to_csv('Crucial_info.csv')
        return df

    def relocate_pixels(self):
        image = self.open_image()
        image_array = np.array(image)
        height, width, _ = image_array.shape
        self.aspect_ratio = height / width

        try:
            pixel_locations = pd.read_csv('Crucial_info.csv')
            df = pixel_locations.copy()
            if len(df) != height * width:
                raise ValueError
        except (FileNotFoundError, ValueError):
            df = self.rotate_and_process()
            pixel_locations = df.iloc[:, [3, 4, 5, 14, 15, 16]]


        # Flatten the image array to map RGB values
        rgb_values = image_array.reshape(-1, 3)

        # Add the RGB values to the DataFrame
        pixel_locations['R_orig'] = rgb_values[:, 0]
        pixel_locations['G_orig'] = rgb_values[:, 1]
        pixel_locations['B_orig'] = rgb_values[:, 2]

        # Prepare data for KDTree
        no_bh_coords = df[['Y_no_bh', 'Z_no_bh']].dropna().to_numpy()

        tree = KDTree(no_bh_coords)

        def find_closest_index(row):
            if pd.isna(row['X_geo_limit']) or pd.isna(row['Y_geo_limit']) or pd.isna(row['Z_geo_limit']):
                return np.nan
            _, idx = tree.query([row['Y_geo_limit'], row['Z_geo_limit']])
            return idx

        closest_indices = pixel_locations.apply(find_closest_index, axis=1)

        pixel_locations['Ray_belong_to'] = closest_indices

        valid_indices = ~pixel_locations['Ray_belong_to'].isna()
        valid_rows = pixel_locations.loc[valid_indices]

        pixel_locations.loc[valid_indices, 'R_mod'] = valid_rows.apply(
            lambda row: pixel_locations.loc[int(row['Ray_belong_to']), 'R_orig'], axis=1
        )
        pixel_locations.loc[valid_indices, 'G_mod'] = valid_rows.apply(
            lambda row: pixel_locations.loc[int(row['Ray_belong_to']), 'G_orig'], axis=1
        )
        pixel_locations.loc[valid_indices, 'B_mod'] = valid_rows.apply(
            lambda row: pixel_locations.loc[int(row['Ray_belong_to']), 'B_orig'], axis=1
        )
        pixel_locations.to_csv('Pixel_information.csv')
        return pixel_locations

    def create_scene_matplotlib(self):

        try :
            pl = pd.read_csv('Pixel_information.csv')
            if len(pl) != self.image_width * self.image_height:
                raise ValueError
        except (FileNotFoundError, ValueError):
            pl = self.relocate_pixels()

        ## Picture modified by BH
        Y_no_bh = pl['Y_no_bh']
        Z_no_bh = pl['Z_no_bh']
        R_mod = pl['R_mod'] / 255.0
        G_mod = pl['G_mod'] / 255.0
        B_mod = pl['B_mod'] / 255.0

        # Combine the RGB values into a single array
        colors = np.array([R_mod, G_mod, B_mod]).T


        # Create the scatter plot
        plt.style.use('dark_background')
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

        # Original image subplot

        ax.scatter(Y_no_bh, -Z_no_bh, c=colors, marker='s')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Show the plot
        plt.show()

    def create_scene_PIL(self):

        try :
            pl = pd.read_csv('Pixel_information.csv')
            if len(pl) != self.image_width * self.image_height:
                raise ValueError
        except (FileNotFoundError, ValueError):
            pl = self.relocate_pixels()

        ## Picture modified by BH
        Y_no_bh = pl['Y_no_bh']
        Z_no_bh = pl['Z_no_bh']
        R_mod = pl['R_mod'] / 255.0
        G_mod = pl['G_mod'] / 255.0
        B_mod = pl['B_mod'] / 255.0

        # Normalize coordinates to fit in the image dimensions
        width = int(max(Y_no_bh) - min(Y_no_bh) + 1)
        height = int(max(Z_no_bh) - min(Z_no_bh) + 1)

        # Normalize coordinates to start from zero
        Y_no_bh = Y_no_bh - min(Y_no_bh)
        Z_no_bh = Z_no_bh - min(Z_no_bh)

        # Create an empty image
        img = Image.new('RGB', (width, height))

        # Combine the RGB values into a single array
        colors = np.array([R_mod, G_mod, B_mod]).T * 255

        # Convert color array to list of tuples
        colors = [tuple(color.astype(int)) for color in colors]

        # Set pixel colors based on coordinates and color values
        for y, z, color in tqdm(zip(Y_no_bh, Z_no_bh, colors), desc="Setting pixel colors", total=len(colors)):
            img.putpixel((int(y), int(z)), color)

        # Save or display the image

        img.show()  # Display the image

if __name__ == "__main__":
    Rs = 5
    D = 10 * Rs  ## Distance of the photon shooter to BH. This is literally the camera position.
    Dv = D - Rs
    resize_length = 1000
    fov_side = 130
    fov_above = 65
    R_isco = 1.5 * Rs  ## Because this is the schwarszchild blackhole

    Blackhole = black_hole(Rs, D, Dv, 'Hubble.png', resize_length, fov_side, fov_above)
    Blackhole.create_scene_matplotlib()




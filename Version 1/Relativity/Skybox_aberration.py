import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pathlib import Path
import numba
# import pygame
import math
from scipy.spatial import KDTree
from scipy.constants import c
import moviepy.editor as mpy
from tqdm import trange

class ConvertToCubemap():
    def __init__(self, image_path, resize : tuple):
        self.image = Image.open(image_path)

        if resize!=None:
            self.image = self.image.resize(resize, resample=Image.LANCZOS)

        self.length, self.height = self.image.size
        ## Check if image_length is not the same
        self.image_array = np.array(self.image)
        if (self.length/self.height) != 2:
            raise ValueError('Image length is not 2 to 1!, Find another image')
        self.image_array = np.array(self.image)


    def rect_to_unitsphere(self):
        azimuth_map, elevation_map, color_map = self.compute_spherical_coordinates(
            self.height, self.length, self.image_array
        )
        return azimuth_map, elevation_map, color_map
    @staticmethod
    @numba.njit
    def compute_spherical_coordinates(height, length, image_array):
        azimuth_map = np.zeros((height, length))
        elevation_map = np.zeros((height, length))
        color_map = np.zeros((height, length, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(length):
                u = x / length
                v = y / height

                # Convert normalized coordinates to azimuth and elevation
                azimuth = (u * 2 * np.pi) - np.pi
                elevation = (v * np.pi) - (np.pi / 2)

                # Store the results
                azimuth_map[y, x] = azimuth
                elevation_map[y, x] = elevation
                color_map[y, x] = image_array[y, x, :3]

        return azimuth_map, elevation_map, color_map

    def aberration_calc(self, beta, angle_map, modify=False):

        if modify:
            angle_map_mod = angle_map * 1.0
            numerator = np.cos(angle_map_mod) - beta
            denom = 1 - beta * np.cos(angle_map_mod)
            observed_angles = np.arccos(numerator / denom)
            # Preserve the sign of the original angles
            observed_angles *= np.sign(angle_map_mod)
            return observed_angles

        angle_map = angle_map
        numerator = np.cos(angle_map) - beta
        denom = 1 - beta * np.cos(angle_map)
        observed_angles = np.arccos(numerator / denom)
        # Preserve the sign of the original angles
        observed_angles *= np.sign(angle_map)
        return observed_angles

    def aberration_modify(self, velocity, azimuth_map, elevation_map, light_speed = c):
        beta = velocity / light_speed
        transformed_az = self.aberration_calc(beta, azimuth_map, modify=False)
        # transformed_el = self.aberration_calc(beta, elevation_map, modify=False)  # Try one of the angles is unchanged.
        transformed_el = elevation_map
        return transformed_az, transformed_el

    def lorentz_factor(self, velocity, light_speed = c, modulate=False, power = 0.43):
        beta = velocity / light_speed
        if modulate:
            factor = np.power( (1 - (beta ** 2)) , power) / np.sqrt( (1 - (beta ** 2)) )
        else:
            factor = 1 / np.sqrt((1 - (beta ** 2)))
        return factor

    # Visualize the spherical projection
    def visualize_spherical_projection(self, azimuth_map, elevation_map, color_map):
        # Convert spherical coordinates to Cartesian for visualization
        x_coords = np.cos(elevation_map) * np.cos(azimuth_map)
        y_coords = np.cos(elevation_map) * np.sin(azimuth_map)
        z_coords = np.sin(elevation_map)

        # Plot the spherical projection using scatter plot
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_coords.flatten(), y_coords.flatten(), z_coords.flatten(), c=color_map.reshape(-1, 3) / 255.0, s=10.0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_title('Spherical Projection')
        ax.view_init(elev=30, azim=30)
        plt.show()

    def test_spherical_projection(self, beta):
        azimuth_map, elevation_map, color_map = self.rect_to_unitsphere()

        # Visualize the original spherical projection
        self.visualize_spherical_projection(azimuth_map, elevation_map, color_map)

        # Apply aberration and visualize the modified spherical projection
        velocity = -beta * c
        transformed_az, transformed_el = self.aberration_modify(velocity, azimuth_map, elevation_map)

        self.visualize_spherical_projection(transformed_az, transformed_el, color_map)

class Doppler_shift(ConvertToCubemap):
    def convert_K_to_RGB(self, colour_temperature):
        tmp_actual = colour_temperature / 100.0
        # red
        if tmp_actual <= 66:
            red = 255
        else:
            tmp_red = 329.698727446 * math.pow(tmp_actual - 60, -0.1332047592)
            red = min(max(int(tmp_red), 0), 255)

        # green
        if tmp_actual <= 66:
            tmp_green = 99.4708025861 * math.log(tmp_actual) - 161.1195681661
            green = min(max(int(tmp_green), 0), 255)
        else:
            tmp_green = 288.1221695283 * math.pow(tmp_actual - 60, -0.0755148492)
            green = min(max(int(tmp_green), 0), 255)

        # blue
        if tmp_actual >= 66:
            blue = 255
        elif tmp_actual <= 19:
            blue = 0
        else:
            tmp_blue = 138.5177312231 * math.log(tmp_actual - 10) - 305.0447927307
            blue = min(max(int(tmp_blue), 0), 255)

        # Remodify the RGB color for colors less than 1000 Kelvin
        if 0 < tmp_actual <= 10.0:
            blue = 0
            red = int(red * math.pow(tmp_actual / 10.0, 1.25))
            green = int(green * math.pow(tmp_actual / 10.0, 1.25))

        return red, green, blue

    def generate_temperature_rgb_lookup(self, start_temp, end_temp, step_size):
        lookup_table = []
        for temp in range(start_temp, end_temp + 1, step_size):
            rgb = self.convert_K_to_RGB(temp)
            lookup_table.append((temp, rgb))
        return lookup_table

    def find_closest_temperature(self, rgb, lookup_table):
        rgb_values = np.array([entry[1] for entry in lookup_table])
        kd_tree = KDTree(rgb_values)
        _, idx = kd_tree.query(rgb)
        return lookup_table[idx][0]


    def test(self):
        # Generate lookup table
        start_temp = 1
        end_temp = 15000
        step_size = 10
        lookup_table = self.generate_temperature_rgb_lookup(start_temp, end_temp, step_size)

        # Plot temperature colors
        for temp, rgb in lookup_table:
            color = np.array(rgb) / 255.0
            plt.plot((temp / 100, temp / 100), (0, 1), linestyle="-", color=color)

        # Show the plot
        plt.show()

        # Example usage: Find the closest temperature for a given RGB value
        example_rgb = (100, 100, 255)
        closest_temp = self.find_closest_temperature(example_rgb, lookup_table)
        print(f'The closest temp is {closest_temp} C')



def wrap_angle_radians(angle, azimuthal=True):
    if azimuthal:
        wrapped_angle = angle % (2 * np.pi)
        if wrapped_angle < 0:
            wrapped_angle += 2 * np.pi
        return wrapped_angle
    else:
        wrapped_angle = angle % np.pi
        if wrapped_angle < 0:
            wrapped_angle += np.pi
        return wrapped_angle

def render_sphere(azimuth_map, elevation_map, color_map, azimuth, elevation, az_fov, el_fov, screen_width, screen_height):
    output = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    half_az_fov = az_fov / 2
    half_el_fov = el_fov / 2

    # Flatten the maps for KDTree
    flat_coords = np.vstack((azimuth_map.ravel(), elevation_map.ravel())).T
    tree = KDTree(flat_coords)

    for screen_y in range(screen_height):
        for screen_x in range(screen_width):
            delta_az = ((screen_x / screen_width) * az_fov) - half_az_fov
            delta_el = ((screen_y / screen_height) * el_fov) - half_el_fov

            az = wrap_angle_radians(azimuth + delta_az, azimuthal=True)
            el = wrap_angle_radians(elevation + delta_el, azimuthal=False)

            dist, index = tree.query([az, el])
            closest_color = color_map.reshape(-1, 3)[index]

            output[screen_y, screen_x] = closest_color

    return output

def save_frames_as_mp4(velocities, azimuth, elevation, az_fov, el_fov, screen_width, screen_height):
    frames = []

    # Path to a common font file on Windows
    font_path = "C:/Windows/Fonts/arial.ttf"
    font = ImageFont.truetype(font_path, 24)
    file_name = Path(r'C:\Users\Bowen\PycharmProjects\General Relativity\Schwarzschild')
    image_path = file_name / 'milkyway.jpg'
    converter = ConvertToCubemap(image_path, None)

    azimuth_map, elevation_map, color_map = converter.rect_to_unitsphere()
    for i in trange(len(velocities), desc="Rendering frames"):
        velocity = velocities[i]

        # Apply aberration based on the current velocity
        transformed_az, transformed_el = converter.aberration_modify(-velocity, azimuth_map, elevation_map)

        # Render the visible part of the sphere using the optimized function
        pixel_data = render_sphere(transformed_az, transformed_el, color_map, azimuth, elevation, az_fov, el_fov,
                                   screen_width, screen_height)

        # Convert the rendered data to an image
        image = Image.fromarray(pixel_data)
        angle = -90
        image = image.rotate(angle, expand=True)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Add velocity text overlay
        draw = ImageDraw.Draw(image)
        text = f"v = {velocity / c:.5f}c"
        draw.text((8, 8), text, font=font, fill=(255, 255, 255))

        # Convert the PIL image to a NumPy array and append to frames
        frames.append(np.array(image))

    # Create a video clip from the frames
    clip = mpy.ImageSequenceClip(frames, fps=10)  # Adjust FPS as needed

    # Write the video file
    clip.write_videofile("spherical_view.mp4", codec="libx264")



def main(video=False):
    # Load image and compute spherical coordinates
    file_name = Path(r'C:\Users\Bowen\PycharmProjects\General Relativity\Schwarzschild')
    image_path = file_name / 'Equirectangular_projection_test.jpg'
    converter = ConvertToCubemap(image_path, None)
    azimuth_map, elevation_map, color_map = converter.rect_to_unitsphere()

    # Rescale azimuth_map from [-pi, pi] to [0, 2*pi]
    azimuth_map = (azimuth_map + 2 * np.pi) % (2 * np.pi)
    elevation_map = elevation_map + np.pi / 2

    # Initial view angles
    azimuth = np.deg2rad(0)
    elevation = np.deg2rad(90)

    # Field of view (in radians)
    el_fov = np.deg2rad(60)  # 60 degrees top-down
    az_fov = np.deg2rad(100)  # 100 degrees left-right

    # Screen dimensions based on FOV
    screen_width = 400  # Adjust width based on desired resolution
    screen_height = int(screen_width * el_fov / az_fov)  # Height to maintain aspect ratio based on FOV

    # Render the visible part of the sphere using the optimized function
    pixel_data = render_sphere(azimuth_map, elevation_map, color_map, azimuth, elevation,
                               az_fov, el_fov, screen_width, screen_height)

    # Plot the rendered image using Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(pixel_data, interpolation='nearest')
    plt.xlabel("Screen Width")
    plt.ylabel("Screen Height")
    plt.show()

if __name__ == "__main__":
    main(video=False)


# def main(video=False):
#     # Initialize Pygame
#     pygame.init()
#
#     # Load image and compute spherical coordinates
#     file_name = Path(r'C:\Users\Bowen\PycharmProjects\General Relativity\Schwarzschild')
#     image_path = file_name / 'Equirectangular_projection_test.jpg'
#     converter = ConvertToCubemap(image_path, (240, 120))
#     azimuth_map, elevation_map, color_map = converter.rect_to_unitsphere()
#     # Initial view angles
#     azimuth = 0.0
#     elevation = 0.0
#
#     # Field of view (in radians)
#     el_fov = np.deg2rad(60)  # 60 degrees top-down
#     az_fov = np.deg2rad(100)  # 100 degrees left-right
#
#     # Screen dimensions based on FOV
#     screen_width = int(800)  # Adjust width based on desired resolution
#     screen_height = int(screen_width * el_fov / az_fov)  # Height to maintain aspect ratio based on FOV
#
#     screen = pygame.display.set_mode((screen_width, screen_height))
#     pygame.display.set_caption("Spherical View")
#
#     # Movement sensitivity
#     acceleration_sensitivity = 0.012
#     deceleration_sensitivity = 0.01
#     azimuth_sensitivity = 0.10
#     elevation_sensitivity = 0.08
#     velocity = 0.0  # Initial velocity
#     max_velocity = 0.9999 * c  # Maximum velocity
#
#     # Font for displaying speed
#     font_path = "C:/Windows/Fonts/arial.ttf"
#     font = pygame.font.Font(font_path, 24)
#
#     # Initialize lists to store velocities
#     acc_vel = []
#     dec_vel = []
#
#     running = True
#     clock = pygame.time.Clock()
#
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#
#         keys = pygame.key.get_pressed()
#         if keys[pygame.K_w]:  # Accelerate
#             acc_sensitivity = acceleration_sensitivity / converter.lorentz_factor(velocity, modulate=True,
#                                                                                   power=0.480407)
#             velocity = min(velocity + acc_sensitivity * c, max_velocity)
#             acc_vel.append(velocity)
#         if keys[pygame.K_s]:  # Decelerate
#             dec_sensitivity = deceleration_sensitivity / converter.lorentz_factor(velocity, modulate=True, power=0.25)
#             velocity = max(velocity - dec_sensitivity * c, 0)
#             dec_vel.append(velocity)
#
#         # Adjust azimuth angle
#         if keys[pygame.K_LEFT]:
#             azimuth -= azimuth_sensitivity
#         if keys[pygame.K_RIGHT]:
#             azimuth += azimuth_sensitivity
#
#         # Adjust elevation angle
#         if keys[pygame.K_UP]:
#             elevation += elevation_sensitivity
#         if keys[pygame.K_DOWN]:
#             elevation -= elevation_sensitivity
#
#         # Clamp elevation to prevent flipping over the top
#         elevation = np.clip(elevation, -np.pi / 2, np.pi / 2)
#
#         # Wrap azimuth angle to stay within the range -pi to pi
#         azimuth = (azimuth + np.pi) % (2 * np.pi) - np.pi
#
#         # Apply aberration based on the current velocity
#         transformed_az, transformed_el = converter.aberration_modify(-velocity, azimuth_map, elevation_map)
#
#         # Render the visible part of the sphere using the optimized function
#         pixel_data = render_sphere(transformed_az, transformed_el, color_map, azimuth, elevation, az_fov, el_fov,
#                                    screen_width, screen_height)
#
#         # Create a Pygame surface from the rendered data
#         sphere_surface = pygame.surfarray.make_surface(pixel_data)
#
#         # Blit the sphere surface onto the main screen
#         screen.blit(sphere_surface, (0, 0))
#
#         # Render the speed text
#         speed_text = font.render(f"v = {velocity / c:.4f}c", True, (255, 255, 255))
#         screen.blit(speed_text, (10, 10))
#
#         # Update the display
#         pygame.display.flip()
#
#         # Cap the frame rate
#         clock.tick(60)
#
#     pygame.quit()
#
#     # Combine acceleration and deceleration lists
#     if video:
#         velocities = acc_vel + dec_vel
#         save_frames_as_mp4(velocities, azimuth, elevation, az_fov, el_fov, screen_width, screen_height)
#
#
# if __name__ == "__main__":
#     main(video=False)
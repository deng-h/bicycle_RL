import pybullet as p
import pybullet_data
import numpy as np
import time


def simulate_lidar(sensor_origin, num_rays, ray_length, angle_range):
    """
    Simulate a LIDAR sensor using ray casting.

    :param sensor_origin: 传感器的原点 [x, y, z]
    :param num_rays: 要投射的射线数量
    :param ray_length: 每条射线的最大长度
    :param angle_range: 射线要覆盖的角度范围（以弧度为单位）
    :return: Distances measured by the rays
    """
    rays_from = []
    rays_to = []

    # Generate ray directions
    angles = np.linspace(-angle_range / 2, angle_range / 2, num_rays)
    for angle in angles:
        dx = ray_length * np.cos(angle)
        dy = ray_length * np.sin(angle)
        dz = 0
        ray_from = sensor_origin
        ray_to = [sensor_origin[0] + dx, sensor_origin[1] + dy, sensor_origin[2] + dz]
        rays_from.append(ray_from)
        rays_to.append(ray_to)

    # Cast rays
    results = p.rayTestBatch(rays_from, rays_to)
    ray_results = [(ray_from, result[3] if result[0] != -1 else ray_to) for ray_from, ray_to, result in zip(rays_from, rays_to, results)]

    return ray_results


def visualize_lidar(ray_results):
    """
    Visualize the LIDAR rays in the PyBullet environment.

    :param ray_results: A list of (start_point, end_point) tuples for each ray
    """
    for ray_from, ray_to in ray_results:
        p.addUserDebugLine(ray_from, ray_to, lineColorRGB=[1, 0, 0], lifeTime=0.1)

# Initialize PyBullet
p.connect(p.GUI)
p.setTimeStep(1. / 24.)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.loadURDF("r2d2.urdf", [0, 0, 0.5])

# Set simulation parameters
sensor_origin = [2, 2, 0.5]
num_rays = 360
ray_length = 10
angle_range = 2 * np.pi

# Run simulation
try:
    while True:
        ray_results = simulate_lidar(sensor_origin, num_rays, ray_length, angle_range)
        visualize_lidar(ray_results)
        p.stepSimulation()

        time.sleep(1. / 24.)
finally:
    p.disconnect()


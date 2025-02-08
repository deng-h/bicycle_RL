import trimesh

def compute_inertia_matrix(mesh_file, mass):
    """
    计算不规则物体的惯性矩阵
    Args:
        mesh_file (str): 3D 网格文件路径（如 STL 或 OBJ）
        mass (float): 物体质量

    Returns:
        inertia (dict): 惯性矩阵信息，包括中心、惯性矩阵、比例缩放。
    """
    # 加载网格文件
    mesh = trimesh.load(mesh_file)

    if not mesh.is_volume:
        raise ValueError("Mesh must be watertight (closed volume) to compute inertia.")

    # 计算体积和密度
    volume = mesh.volume
    density = mass / volume  # 假设质量分布均匀

    # 计算质心
    center_of_mass = mesh.center_mass

    # 计算相对于质心的惯性矩阵
    inertia_tensor = mesh.moment_inertia * density  # 质量密度缩放

    return {
        "center_of_mass": center_of_mass,
        "inertia_matrix": inertia_tensor,
        "volume": volume,
        "density": density,
    }

# 示例：计算一个不规则物体的惯性矩阵
mesh_file = "D:\\data\\1-L\\9-bicycle\\bicycle-rl\\bicycle_dengh\\resources\\small_bicycle\\files\\zhong_jian_che_ti.stl"  # 替换为实际的 STL 文件路径
mass = 0.05  # 假设质量为 1.0 千克

result = compute_inertia_matrix(mesh_file, mass)

print("Center of Mass:", result["center_of_mass"])
print("Inertia Matrix (relative to center of mass):\n", result["inertia_matrix"])
print("Volume:", result["volume"])
print("Density:", result["density"])

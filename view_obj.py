import open3d as o3d

mesh = o3d.io.read_triangle_mesh("output/25_09_24-13_48_16_selfcon_ttc/obj_models/scale100.obj")
o3d.visualization.draw_geometries([mesh])
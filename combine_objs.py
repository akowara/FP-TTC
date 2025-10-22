import open3d as o3d
import time

objs = [o3d.io.read_triangle_mesh(f"output/25_09_24-14_11_12_selfcon_ttc/obj_models/scale{i*5}.obj") for i in range(40)]

for obj in objs:
    obj.compute_vertex_normals()

vis = o3d.visualization.Visualizer()
vis.create_window()

for obj in objs:
    vis.add_geometry(obj)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.5)
    vis.remove_geometry(obj)

vis.destroy_window()
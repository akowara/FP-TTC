import open3d as o3d
import os
import time
import natsort
import imageio
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folderPath', default='', type=str,
                    help='where the depth maps are stored')

args = parser.parse_args()
folder_path = args.folderPath

obj_files = [f for f in os.listdir(folder_path) if f.endswith('.obj')]
obj_files = natsort.natsorted(obj_files)
# mesh = o3d.io.read_triangle_mesh("output/25_09_24-13_48_16_selfcon_ttc/obj_models/scale100.obj")
# o3d.visualization.draw_geometries([mesh])

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Animation", width=960, height=720)
mesh = o3d.io.read_triangle_mesh(os.path.join(folder_path, obj_files[0]))
mesh.compute_vertex_normals()
vis.add_geometry(mesh)


frames = []
for file in obj_files:
    mesh_new = o3d.io.read_triangle_mesh(os.path.join(folder_path, file))
    mesh.vertices = mesh_new.vertices
    mesh.triangles = mesh_new.triangles
    mesh.compute_vertex_normals()
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    img = vis.capture_screen_float_buffer(False)
    img = (255 * np.asarray(img)).astype(np.uint8)
    frames.append(img)

os.mkdir(f"output/{folder_path.split('/')[1]}/animation")
imageio.mimsave(f"output/{folder_path.split('/')[1]}/animation/scene_animation.mp4", frames, fps=10)
import numpy as np
import os
import argparse
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('--eta_path', default='', type=str,
                    help='where the MID maps are stored')

parser.add_argument('--fps', default=10, type=int,
                    help='FPS of camera')

args = parser.parse_args()
directory = args.eta_path
fps = args.fps
delta_t = 1/fps

npy_files = natsorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])
new_directory = f'{directory}/depth_maps'

try:
    # Create the directory
    os.mkdir(new_directory)
except FileExistsError:
    print(f"Directory '{new_directory}' already exists.")

for file in npy_files:
    data = np.load(file)
    data = delta_t/(1-data)
    np.save(os.path.join(new_directory, file.split('_')[-3].split('/')[1]+'_depth.npy'), data)
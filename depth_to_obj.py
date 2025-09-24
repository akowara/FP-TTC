
import numpy as np
from PIL import Image
from pathlib import Path
import os
from DepthMapVisualizer import DepthToObj
import argparse
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('--depthPath', default='', type=str,
                    help='where the depth maps are stored')

args = parser.parse_args()
directory = args.depthPath

image_files = natsorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')])

images = [Image.open(img) for img in image_files]

grayscale_path = Path(directory) / "grayscale"
grayscale_path.mkdir(parents=True, exist_ok=True)

for file in image_files:
    image = Image.open(file)
    if image.mode == 'RGB':
        
        gray = image.convert('L')

        arr_8bit = np.array(gray)
        arr_16bit = (arr_8bit.astype(np.uint16)) * 257  # 255*257=65535
        img_16bit = Image.fromarray(arr_16bit, mode='I;16')
        img_16bit.save(f"{grayscale_path}/{file.split('/')[-1]}")

gs_images = natsorted([os.path.join(grayscale_path, f) for f in os.listdir(grayscale_path) if f.endswith('.png')])

args_2 = DepthToObj.parse_args()
useMat = args_2.texturePath != ''

model_path = Path(directory) / "obj_models"
model_path.mkdir(parents=True, exist_ok=True)
# file = gs_images[0]
for file in gs_images:

    args_2.depthPath = file
    args_2.objPath = f"{model_path}/{file.split('/')[-1].split('.')[0]}.obj"

    DepthToObj.create_obj(args_2.depthPath, args_2.depthInvert, args_2.objPath, args_2.mtlPath, args_2.matName, useMat)
    print("FINISHED")



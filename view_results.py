import numpy as np
from PIL import Image

data = np.load('output/25_09_24-13_48_16_selfcon_ttc/100_eta_out.npy')
print("Type:", type(data))
print("Shape:", data.shape)
print("Dtype:", data.dtype)
print("Min:", np.min(data))
print("Max:", np.max(data))
print("Mean:", np.mean(data))
print(data[:10])

# img = Image.open('output/25_09_22-18_02_25_selfcon_ttc/scale0.png')
# img_array = np.array(img)/255.0
# print("Image pixel values:")
# print("Type:", type(img_array))
# print("Shape:", img_array.shape)
# print("Dtype:", img_array.dtype)
# print("Min:", np.min(img_array))
# print("Max:", np.max(img_array))
# print("Mean:", np.mean(img_array))
# print(img_array[:10])
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

data = np.load('output/25_10_23-13_21_56_selfcon_ttc/120_eta_out.npy')
# data = (data- data.min()) / (data.max() - data.min())
bl = 0.5327254279298227
K= [[721.5377,   0,     609.5593],
    [  0,     721.5377, 172.854 ],
    [  0,       0,       1  ]]
depth = (0.1/(1-data))
print(depth)
print("Type:", type(data))
print("Shape:", data.shape)
print("Dtype:", data.dtype)
print("Min:", np.min(data))
print("Max:", np.max(data))
print("Mean:", np.mean(data))
disp = (bl*K[0][0])/depth
print(disp.shape)



# img = Image.open('output/25_10_23-13_21_56_selfcon_ttc/scale120.png')
# plt.figure(figsize=(12,4)); plt.imshow(img); plt.axis('off')
# plt.show()
# img_array = np.array(img)
# print("Image pixel values:")
# print("Type:", type(img_array))
# print("Shape:", img_array.shape)
# print("Dtype:", img_array.dtype)
# print("Min:", np.min(img_array))
# print("Max:", np.max(img_array))
# print("Mean:", np.mean(img_array))
# print(img_array)

# normalized_data = (img_array- img_array.min()) / (img_array.max() - img_array.min())
# print("NORMLA",normalized_data[:10])


import cv2
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np

save_path = '../../../images'

###########################################
# Trilinear Interpolation
###########################################
image_path = "../vimeo_septuplet/sequences_scale_4/00011/0001"
image1 = cv2.imread(image_path + "/im1.png")
image3 = cv2.imread(image_path + "/im3.png")

image1 = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
image3 = torch.from_numpy(image3).permute(2, 0, 1).unsqueeze(0).float() / 255.0
image1 = image1[:, :, :, :96] # crop width to 96
image3 = image3[:, :, :, :96] # crop width to 96

B, C, H, W = image1.shape

images = torch.cat([image1, image3], dim=0).unsqueeze(0)

images = images.permute(0, 2, 1, 3, 4)
images = F.interpolate(images, (3, H, W), mode='trilinear', align_corners=False)
images = images.permute(0, 2, 1, 3, 4)

images = images.squeeze()
images = torch.cat([images[0, :, :, :], images[1, :, :, :], images[2, :, :, :]], dim=1).permute(1, 2, 0)
images = (images * 255).byte().numpy()

# plt.imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
# plt.show()

###########################################
# Residual
###########################################
image_path = "../vimeo_septuplet/sequences_scale_4/00011/0001"
gt = cv2.imread(image_path + "/im1.png")
interpolated = cv2.imread(image_path + "/im2.png")

residual = gt - interpolated

cv2.imwrite(save_path + "/residual.png", residual)

# plt.imshow(residual)
# plt.show()

###########################################
# Queries
###########################################
image_path = "../vimeo_septuplet/sequences_scale_4/00011/0001"
image1 = cv2.imread(image_path + "/im1.png")
image2 = cv2.imread(image_path + "/im2.png")
image3 = cv2.imread(image_path + "/im3.png")

image1 = torch.from_numpy(image1).float()
image2 = torch.from_numpy(image2).float()
image3 = torch.from_numpy(image3).float()

query_1 = torch.zeros(image2.shape).numpy()
query_2 = ((image2[:, :, :] - (image1[:, :, :] + image3[:, :, :])/2)).abs().numpy()
query_3 = torch.zeros(image2.shape).numpy()

cv2.imwrite(save_path + "/query1.png", query_1)
cv2.imwrite(save_path + "/query1.5.png", query_2*255)
cv2.imwrite(save_path + "/query2.png", query_3)

# plt.imshow(cv2.cvtColor(query_2, cv2.COLOR_BGR2RGB))
# plt.show()

###########################################
# Params vs PSNR ( mine vs others )
###########################################

x = [ # PSNR
    33.51, # TMNet
    33.30, # RSTT-S
    30.254721030049133, # Mine
]
y = [ # Params
    12.26, # TMNet
    4.49, # RSTT-S
    3.441354, # Mine
]

labels = [
    ('TMNet', 'right', 'center'),
    ('RSTT-S', 'left', 'center'),
    ('Our Model', 'left', 'center')
]

plt.figure(figsize=(12, 9))
plt.scatter(x, y)
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.text(xi + ((1 if labels[i][1] == 'left' else -1) * 0.02), yi, labels[i][0], ha=labels[i][1], va=labels[i][2], fontsize=12, color='black')

plt.title('PSNR vs Parameters')
plt.xlabel('Peak Signal to Noise Ratio (PSNR) (db)', fontsize=13)
plt.ylabel('Number of Parameters (Million)', fontsize=13)
plt.savefig(save_path + '/psnr_vs_params.png')
# plt.show()

###########################################
# PSNR vs FPS ( mine vs others )
###########################################

###########################################
# PSNR vs FPS ( ablation )
###########################################
x = [ # PSNR
    30.703070175876, # Initial model
    31.427182587761674, # Positional Encoding
    31.876908464544723, # Context Compression
]
y = [ # FPS
    21.7724029186, # Initial model
    21.7248717798, # Positional Encoding
    14.6821839248, # Context Compression
]

labels = [
    ('Basic Model', 'left', 'center'),
    ('Basic + Positional Encoding', 'left', 'center'),
    ('Basic + PE + Context Compression', 'right', 'center')
]

plt.figure(figsize=(12, 9))
plt.scatter(x, y)
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.text(xi + ((1 if labels[i][1] == 'left' else -1) * 0.02), yi, labels[i][0], ha=labels[i][1], va=labels[i][2], fontsize=12, color='black')

plt.title('Ablation Study: PSNR vs FPS')
plt.xlabel('Peak Signal to Noise Ratio (PSNR) (db)', fontsize=13)
plt.ylabel('Output FPS', fontsize=13)
plt.savefig(save_path + '/psnr_vs_fps.png')
# plt.show()
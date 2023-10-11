import cv2
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F

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

plt.imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
plt.show()

###########################################
# Params vs PSNR ( mine vs others )
###########################################

###########################################
# PSNR vs FPS ( mine vs others)
###########################################

###########################################
# Params vs PSNR ( ablation )
###########################################
x = [ # Params
    3.465, # initial model
    3.465, # positional encoding
]
y = [ # PSNR
    30.712,
    31.427,
]
plt.scatter(x, y)
plt.show()

###########################################
# PSNR vs FPS ( ablation )
###########################################
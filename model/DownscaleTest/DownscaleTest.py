import cv2
import matplotlib.pyplot as plt
import numpy as np

def calculate_psnr(original, downsampled):
    mse = np.mean((original - downsampled) ** 2)
    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

img = cv2.imread('./test4.png')
img = img[:, :, [2, 1, 0]] # swap the channels of the image
scale = 4

interpolations = [
    cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,
    cv2.INTER_CUBIC,
    cv2.INTER_AREA,
    cv2.INTER_LANCZOS4,
    cv2.INTER_LINEAR_EXACT
]

fig, axs = plt.subplots(3, 3, figsize=(18,16), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0, hspace=0)

axs[0][0].axis('off')
axs[0][1].axis('off')
axs[0][2].axis('off')
axs[0][1].imshow(img)

for i in range(len(interpolations)):
    scaled_img = cv2.resize(img, None, fx=1./scale, fy=1./scale, interpolation=interpolations[i])
    scaled_img = np.repeat(np.repeat(scaled_img, scale, axis=0), scale, axis=1)
    axs[1+i//3][i%3].imshow(scaled_img)
    axs[1+i//3][i%3].axis('off')
    
    psnr = calculate_psnr(img, scaled_img)
    text = f'PSNR: {psnr:.2f}'
    axs[1+i//3][i%3].text(scaled_img.shape[1]//2, scaled_img.shape[0]-10, text, ha='center', color='white', fontsize=8, bbox={'facecolor': 'black', 'alpha': 0.8, 'pad': 2})

plt.show()
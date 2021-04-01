import numpy as np
import random
from PIL import Image
import fastai.vision.augment

import cv2
from matplotlib import pyplot as plt

import albumentations as A

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)

# npimg = cv2.imread('../image2.jpg')
# npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)

RESOLUTION=400
anorm = A.Normalize()
asharp = A.Sharpen(p=1.0, alpha=(0.1,0.3), lightness=(0.3, 0.9))
afpca = A.FancyPCA(p=1.0, alpha=0.5)
ars4 = A.Resize(RESOLUTION*4, RESOLUTION*4, always_apply=True)
ars = A.Resize(RESOLUTION, RESOLUTION, always_apply=True)
minsize = 400# default
arrc = A.RandomResizedCrop(always_apply=True, p=1.0, height=minsize, width=minsize, scale=(0.01, 0.01), interpolation=0)#A.RandomResizedCrop(size=minsize, p=1.0, min_scale=0.2, ratio=(1, 1))

im1,im2,im3,im4 = None,None,None,None




def create_pip(img):
    xdim = img.shape[0]
    
    # Create PIPs for the GPUs viewing pleasure ONLY if it's representative of a leaf
    for i in range(5):  # only 5 tries, otherwise, stick with it
        pip = arrc(image=img)['image']
        
        if np.mean(pip[1]) < (np.mean(pip[0]) +6): continue   # green > red + 6
        if np.mean(pip[1]) < (np.mean(pip[2]) +10): continue   # green > blue + 6
            
        if np.mean(pip) < 70: continue  # don't accept if too dark
        if np.mean(pip) > 230: continue   # don't accept if too bright
            
        if np.mean(pip[1]) < 90: continue  # specifically green should be midrange
        if np.mean(pip[1]) > 160: continue
            
        # otherwise break out
        break
            
            
    # centralize the image tones somewhat (not exactly)
    if np.mean(pip) < 110: pip = cv2.add(pip, 118 - round(np.mean(pip)))      
    if np.mean(pip) > 150: pip = cv2.subtract(pip, int(round(np.mean(pip) - 140)))
            
    # Sharpen PIPs to counter blowing them up @TODO change the parameters instead of 2x-ing
    pip = asharp(image=pip)['image']
    pip = asharp(image=pip)['image']

    
    return pip





def init_aug(img, resolution=400):
    
    npimg = ars4(image=img, height=RESOLUTION*4, width=RESOLUTION*4)['image']      # upsample for overhead 
    npimg = afpca(image=npimg)['image']     # apply some fancy normalization
    
    minsize = round(npimg.shape[0]/4)        # choose width based on upsampled X dimensions


    # Create the PIPs
    im1 = create_pip(npimg)
    im2 = create_pip(npimg)
    im3 = create_pip(npimg)
    im4 = create_pip(npimg)
    
    mean1g = round(np.mean(im1[1]))
    mean2g = round(np.mean(im2[1]))
    mean3g = round(np.mean(im3[1]))
    mean4g = round(np.mean(im4[1]))

    print(f'im1 mean green:  {mean1g} avg : {round(np.mean(im1[0])/2 + np.mean(im1[2])/2)}')
    print(f'im2 mean green:  {mean2g} avg : {round(np.mean(im2[0])/2 + np.mean(im2[2])/2)}')
    print(f'im3 mean green:  {mean3g} avg : {round(np.mean(im3[0])/2 + np.mean(im3[2])/2)}')
    print(f'im4 mean green:  {mean4g} avg : {round(np.mean(im4[0])/2 + np.mean(im4[2])/2)}')

    # Re-integrate the PIP images to the top of the image which would be likely otherwise be
    # either further away or mostly sky
    i = Image.fromarray(npimg)
    i.paste(Image.fromarray(im1), (0,0))
    i.paste(Image.fromarray(im2), (minsize,0))
    i.paste(Image.fromarray(im3), (minsize*2,0))
    i.paste(Image.fromarray(im4), (minsize*3,0))
    npimg = np.array(i)
    
    
  
    npimg = ars(image=npimg)['image']   # bring it back down to a regular size the GPU can handle
    
    return npimg


# npimg = init_aug(npimg, 400)
# visualize(npimg)
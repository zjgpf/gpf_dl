# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import random
from PIL import Image
from skimage import img_as_float
from skimage.util import random_noise
import scipy.misc
import hashlib
import calendar
import glob
import re
import pdb
from tqdm import tqdm

def img_generator(num_of_imgs, length, output_dir='./tmp_img/', noise_sigma=0.155, normal_mean=285 ):
    digit_imgs = []
    digit_imgs_pad = []
    path_origin = "origin_num/"
    path_padding = "pad_num/"
    for f in os.listdir(path_origin):
        if not f.startswith('tri'): continue
        num = f.split('_')[1][0]
        digit_imgs+=[(Image.open(os.path.join(path_origin,f)),int(num))]
    digit_imgs = sorted(digit_imgs, key = lambda x: x[1])

    # padding images to same size
    bank_img = Image.open('origin_num/blank.jpg')
    bank_img = bank_img.resize((28,50))
    box = (4,7,25,41)
    for v in digit_imgs:
        digit_img =v[0].resize((21,34))
        bank_img.paste(digit_img,box)
        bank_img.save(path_padding+'pad_%s.jpg'%int(v[1]))

    for f in os.listdir(path_padding):
        if not f.startswith('pad'): continue
        num = f.split('_')[1][0]
        digit_imgs_pad+=[(Image.open(os.path.join(path_padding,f)),int(num))]

    
    digit_imgs_pad = sorted(digit_imgs_pad, key = lambda x: x[1])
    digit_imgs_pad = [v[0] for v in digit_imgs_pad]
    
    normal_dis = np.random.normal(normal_mean, 10, num_of_imgs)

    for img_index in tqdm(range(num_of_imgs)):
        num = []
        for i in range(length):
            num += [random.randint(0,9)]
        print(num)
        
        digit_imgs_comb = [digit_imgs_pad[i] for i in num]
        digit_imgs_comb = np.hstack((np.asarray(v) for v in digit_imgs_comb))
        digit_imgs_comb = Image.fromarray(digit_imgs_comb)
        digit_imgs_comb = digit_imgs_comb.resize((int(normal_dis[img_index]),50))
        digit_imgs_comb = img_as_float(digit_imgs_comb)
        digit_imgs_comb = random_noise(digit_imgs_comb, var=noise_sigma**2)
        hash_digest = hashlib.md5(digit_imgs_comb.tostring()).hexdigest()
        num = [str(i) for i in num]
        img_name = ''.join(num)+'_'+str(digit_imgs_comb.shape[0])+'_'+str(digit_imgs_comb.shape[1])+'_'+hash_digest+'.jpg'
        scipy.misc.imsave(os.path.join(output_dir,img_name), digit_imgs_comb)
       
if __name__ == '__main__':
    img_generator(100,12)
    

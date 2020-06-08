# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 03:26:11 2020

@author: Anirudh
"""


from PIL import Image
import glob

for path in glob.glob(r"C:\Users\Anirudh\Desktop\ssd_keras-master\GR\data\train\*"):
    im=Image.open(path)
    im=im.resize((480,300))
    a=path.split('//')
    im.save(a[-1],dpi=(72,72))
    
for path in glob.glob(r"C:\Users\Anirudh\Desktop\ssd_keras-master\GR\data\val\*"):
    im=Image.open(path)
    im=im.resize((480,300))
    a=path.split('//')
    im.save(a[-1],dpi=(72,72))